# GPULlama3 dtype-seam refactor — scoping

Prereq for the flagship (native low-bit/FP4 GPU matmul). Goal: make a new quant format
**data, not a new class matrix**. Scoped against `~/GPULlama3.java` @ `readme-vllm-refresh`.

## The tax today (measured)

Dtype is a hard-coded axis crossed with model × kind × mode:

- `ForwardPlanFactory.create(GGMLType, ExecutionMode, State, Model)` — `switch(quantization)`
  → `createFP16Plan` / `createQ8_0Plan`, each then `switch(modelType)` over 7 models.
- `tornadovm/plan/components/{fp16,q8_0}/` — 7 `*PlanComponents` per dtype (14).
- `tornadovm/layers/type/{fp16,q8_0}/` — **32** near-duplicate layer classes
  ({FFN, Logits} × {base, decode, prefill} × models).
- `tensor/tornado/` — `FP16TornadoTensor`, `Q8_0TornadoTensor` (no Q4/FP4).

**Adding Q4_K or FP4 as a new `type/*` tree ≈ +7 PlanComponents + ~16 layer classes +
factory branch (~23 files), each a copy-paste of the Q8_0 versions.** Do that twice
(Q4_K, FP4) and it's ~46 hand-maintained near-dups.

## What actually differs (fp16 vs q8_0, same model)

`diff LlamaFP16FFNLayers LlamaQ8_0FFNLayers` — the task-graph skeleton is identical
(RMS reduce → qkv proj → rope+kv → attention → out proj → ffn rms → gate/up → down).
Only three things vary, all dtype-local:

1. **Weight accessor**: `weights.wqLayered[l].asHalfFloatArray()` → `.asByteArray()`.
2. **Kernel method ref** per fused op:
   - rms apply: `mapContextWithQuantize` (task `attn_rms_apply_fp16`) → `reductionOneBlock2WithLayer` (`attn_rms_apply`)
   - qkv: `fusedQKVMatmulX` → `fusedQKVMatmulQ8`
   - out/down matvec: `matrixVectorGenericWithResidual` → `matrixVectorGenericWithResidualQ8_0Byte`
   - gate/up: `fusedRmsNormFFNGateUp` → `fullyFusedRmsNormFFNGateUpQ8`
3. **Normalized-activation buffer**: fp16 uses `state.wrapXbFP16`; q8 uses `state.wrapXb`.

Model-specific structure (Qwen3 QK-norm, Phi3 rope, Granite residual scale, flash-vs-parallel
attention) is on the **model axis, not the dtype axis** — fp16 and q8 of the *same* model share
it. So `model-template × dtype-scheme` is a clean factorization: **M×D classes → M + D**.

## Seam design: `QuantScheme`

An object that *emits task-graph tasks* (not just returns an accessor) — the typed
`.task(name, methodRef, typedArgs...)` calls stay encapsulated so per-dtype type differences
never leak into the model template.

```java
interface QuantScheme {
    GGMLType type();

    // Each emits one fused op onto tg, picking kernel ref + weight accessor + task name.
    void emitRmsNormApply (TaskGraph tg, KernelContext ctx, State s, Configuration cfg);
    void emitQkvProjection(TaskGraph tg, KernelContext ctx, State s, TornadoWeights w, int layer, Configuration cfg);
    void emitAttnOutProj  (TaskGraph tg, KernelContext ctx, State s, TornadoWeights w, int layer, Configuration cfg);
    void emitFfnGateUp    (TaskGraph tg, KernelContext ctx, State s, TornadoWeights w, int layer, Configuration cfg);
    void emitFfnDown      (TaskGraph tg, KernelContext ctx, State s, TornadoWeights w, int layer, Configuration cfg);

    FloatArray normalizedActivation(State s);   // wrapXb vs wrapXbFP16
    void addWorkerGrids(GridScheduler sched, int layer, Configuration cfg); // dtype-varying task names
}
```

Implementations: `FP16Scheme`, `Q8_0Scheme` (existing behavior), later `Q4KScheme`,
`FP4Scheme`. Weight-transfer lists (`transferToDevice`/`consumeFromDevice`) also branch on the
accessor set → fold into the scheme (`weightArraysFor(w, layer)`).

One generic `TransformerFFNLayers<W extends TornadoWeights, C extends Configuration>`
takes a `QuantScheme` + optional **model hooks** for the model-axis differences:

```java
abstract class TransformerFFNLayers<W,C> extends AbstractTransformerLayerTaskGraphs<W,C> {
    protected final QuantScheme scheme;
    protected TaskGraph createFFNLayerTaskGraph(int l) {
        var tg = newLayerGraph(l);
        scheme.emitRmsNormApply(tg, ctx, state, config);
        scheme.emitQkvProjection(tg, ctx, state, weights, l, config);
        emitRopeAndKvCache(tg, l);          // model hook (Phi3 override)
        emitAttention(tg, l);               // model hook (flash vs parallel; Qwen3 QK-norm)
        scheme.emitAttnOutProj(tg, ctx, state, weights, l, config);
        emitFfnRmsReduce(tg, l);
        scheme.emitFfnGateUp(tg, ctx, state, weights, l, config);
        scheme.emitFfnDown(tg, ctx, state, weights, l, config);
        return finishLayerGraph(tg, l);
    }
}
```

Model subclasses (`LlamaFFNLayers`, `Qwen3FFNLayers`, …) override only the hooks — one per
model, dtype-agnostic. Logits + decode/prefill variants get the same treatment (mode-template
× dtype-scheme).

## Recommended normalization (shrinks the diff before abstracting)

Unify the trivially-divergent bits so the scheme surface is smaller:
- Rename `attn_rms_apply_fp16` → `attn_rms_apply` uniformly (worker-grid task name).
- Converge on one normalized-activation buffer name where the pipeline allows, else expose via
  `scheme.normalizedActivation()`.

## Phasing (each step compiles + passes regression independently)

1. **Extract `QuantScheme` + `FP16Scheme`/`Q8_0Scheme`** from the existing Llama classes; make
   `LlamaFFNLayers` generic over the scheme. Prove byte-identical logits vs current on Llama-3.2-1B
   (F16 and Q8_0). *No new behavior.*
2. **Migrate remaining 6 models** to the generic template + hooks; delete the 32 `type/*` dups
   → ~7 model templates + 2 schemes. Same for Logits + decode/prefill.
3. **Collapse `ForwardPlanFactory`**: dtype switch becomes `schemeFor(quantization)`; model switch
   stays. `plan/components/{fp16,q8_0}` merge to one set parameterized by scheme.
4. **Add `Q4KScheme` (+`Q4_KTornadoTensor`)** — first proof the seam works: a new format is one
   scheme + one tensor + one factory line, zero new layer classes.
5. Hand off to flagship: `FP4Scheme` (MXFP4/NVFP4) rides the same seam once the TornadoVM
   int8/FP4 MMA intrinsics land.

## Effort / risk

- Steps 1–3 are a **mechanical, behavior-preserving** refactor; risk is regression, fully
  covered by per-layer logit equality vs the pre-refactor build (deterministic decode).
- Net line count *drops* (32→~7 layer classes). Reviewable as one PR per phase.
- Real friction = TornadoVM task-name / buffer-lifetime semantics (the `consumeFromDevice`
  predecessor-graph-name subtlety already documented in `LlamaFP16FFNLayers`); the scheme must
  reproduce those exactly. Keep the transfer/consume lists inside the scheme so they stay in sync
  with the accessor set.
- Blocks nothing else; unblocks Q4_K + FP4 + any future format at O(1) files each.

## Files (primary)

- `tornadovm/layers/AbstractTransformerLayerTaskGraphs.java` (add scheme + hooks) — base
- new `tornadovm/layers/scheme/{QuantScheme,FP16Scheme,Q8_0Scheme}.java`
- `tornadovm/layers/type/{fp16,q8_0}/*` — collapse to `tornadovm/layers/{model}FFNLayers.java`
- `tornadovm/plan/ForwardPlanFactory.java` + `plan/components/{fp16,q8_0}/*` — merge
- `tensor/tornado/TornadoTensor.java` (already has `getQuants`/`getScales` seams) + new tensors
- kernels stay put (`TransformerComputeKernelsLayered`); schemes just reference them
