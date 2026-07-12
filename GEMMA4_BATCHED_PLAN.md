# Gemma 4 batched-decode — port plan & status

Porting the batched-decode engine to Gemma 4 (Q8_0 path, so the existing Q8 tensor-core GEMMs
`gemmMMAQ8` / `gemmMMAQKVQ8` / `gemmMMAGateUpQ8` handle the projections). Every non-projection
op in the single-token `Gemma4Q8_0FFNLayers` (~25 tasks/layer) needs a **batched** variant that
processes B rows. Each kernel is **microbench-validated bit-exact vs its single-token reference**
before assembly (the safe path — the full forward isn't testable until near-complete).

## Per-layer op sequence to batch (from `Gemma4Q8_0FFNLayers`)

```
 attn pre-norm (reduce+apply)  → wrapXb
 q_proj (Q8 GEMM) → q_norm (per-head RMS)
 [own-KV layers] k_proj→k_norm, v_proj→v_norm, rope_and_cache (NEOX + KV write)
 [reuse-KV layers] rope_q_only
 attention (sliding-window / full, scale 1.0)
 wo_proj (Q8 GEMM) → post-attn norm+residual → wrapX
 ffn pre-norm → ffn_gate_up (GeGLU Q8) → ffn_down (Q8 GEMM) → post-ffn norm+residual
 PLE: ple_gate_proj → gelu·mul → ple_proj → post-ple norm+residual
 [optional] layer_output_scale
 layer-0 setup: scale_embedding(√dim), ple_model_proj, ple_proj_scale_norm, ple_merge
 final: RMS + logits GEMM + logit softcap (tanh)
```

## Kernel checklist (`Gemma4BatchDecodeKernels`)

- [x] **batched sliding-window / full attention**, scale 1.0, per-slot KV, FP16 out —
  `batchedGemmaDecodeAttentionFP16Out` (validated bit-exact, windowed + full; `GemmaBatchedAttentionBench`)
- [x] batched **NEOX RoPE + per-slot KV write** (`batchedGemmaDecodeRopeNeox`) + **RoPE-Q-only**
  (`batchedGemmaDecodeRopeQOnly`) — validated (Q/K/V maxRel 7e-5; `GemmaBatchedRopeNormBench`)
- [x] batched **per-head Q/K RMSNorm** (`batchedGemmaPerHeadRmsNorm`) + **V norm**
  (`batchedGemmaPerHeadRmsNormNoWeight`) — validated bit-exact
- [x] batched **pre-norm apply** (`batchedGemmaApplyRmsNorm`) + **norm+residual**
  (`batchedGemmaRmsNormApplyWithResidual`) — validated; reduce via existing `batchedRmsReduceParallel`
- [x] batched **GeGLU** gate/up (`batchedGemmaGeGLUPacked`, gelu over packed Q8 gate/up) — validated
- [x] elementwise `scaleInPlace` / `addAndScale` / `scaleInPlaceFromTensor` — reuse `Gemma4Kernels` as-is (flat, size = B·dim)
- [x] batched **PLE** tasks: `batchedGemmaPleGateGeluMul` + `batchedGemmaPleProjScaleAndNormalize` — validated bit-exact
- [x] **logit softcap** — skipped for greedy (softcap is monotonic → argmax invariant; on-device argmax unaffected)

Projections reuse the existing Q8 MMA GEMMs (`gemmMMAQ8`, `gemmMMAQKVQ8`, `gemmMMAGateUpQ8`).

## Structural handling

- **Per-layer head/FFN dims** — one TaskGraph per layer already bakes each layer's dims as
  constants (as the single-token path does); no extra work.
- **Sliding-window vs full** — the attention kernel takes `windowSize` (full layers pass
  `≥ contextLength`); the per-layer graph passes the layer's value.
- **Shared-KV layers** — reuse-KV layers skip K/V proj + KV write and RoPE-Q-only; their
  attention reads the KV region of `kvReuseLayer(layer)`. Batched: pass that layer's per-slot
  KV base instead of the current layer's.
- **PLE** — per-layer-embedding contribution mixed in each layer; `perLayerInputs` computed once
  at layer 0 (host-gathers the per-token per-layer-embedding row into a batch buffer).
- **Weights** — Q8_0 for the main projections; PLE projections may be F32/F16 (dispatch per tensor).

## Engine

Once the kernels land: a `Gemma4Q8LayersBatchDecodeMMA` layer graph (mirrors the single-token
task order, batched) + `Gemma4State`-backed batch buffers + dispatch in `BatchedDecodeEngine`
(`config instanceof Gemma4Configuration`). The serving features (continuous / paging / prefix /
on-device sampling) are model-agnostic and then apply unchanged.

## Status

**All 10 new batched kernels ported + validated bit-exact** (attention, NEOX RoPE ×2,
per-head norm ×2, GeGLU, RMSNorm-apply, norm+residual, PLE ×2); projections reuse the existing
Q8 MMA GEMMs; elementwise ops reuse `Gemma4Kernels` as-is; softcap skipped for greedy. Each has
a passing microbench (`GemmaBatched{Attention,RopeNorm,FFN,Ple}Bench`).

**Kernels are complete and assembly-ready.** The target model's shape drove two extra kernel
fixes, both done + validated:
- **E2B geometry** (probed): dim 1536, 35 layers, nHeads 8, **nHeadKv 1**, headDim **256 (swa) /
  512 (full)**, 20 shared-KV layers, sliding window 512, nEmbdPerLayer 256, ffn 6144, vocab
  **262144**. Attention rewritten to handle **headDim ≤ 512** (4 register accumulators, 8×512
  tiles) — re-validated bit-exact at 256 and 512.
- **PLE weights are F32** (`perLayerInpGate`/`perLayerProj`) and F16 (`perLayerModelProj`); the
  main projections + `wcls` are Q8_0. Added `batchedMatVecF32` / `batchedMatVecF16` for the PLE
  projections (no MMA path for F32) and `batchedGemmaApplyRmsNormFP16` for the Q8-GEMM inputs.
- **KV addressing** known: per-slot stride = `totalCacheElements` (single-seq, dedup'd), per-layer
  base = `cacheLayerBaseOffset[l]` (reuse layers alias their source); attention/rope take
  `(slotStride, layerBaseOff)`.

**Remaining = engine assembly (the first end-to-end test point):** a `Gemma4BatchedDecodeEngine`
mirroring the single-token task order batched — embed·√dim + layer-0 PLE setup (host-gather the
per-token per-layer-embedding rows into a batch buffer), 35 layer graphs (Q8 MMA q/k/v/o/gate-up/
down + the validated batched kernels; own-KV vs reuse-KV branch; swa/full freq tables + window),
final RMS + Q8 logits GEMM + on-device argmax — then debug greedy vs the single-token reference.
This is dense integration but every kernel it needs is validated; it is de-risked, not open-ended.
