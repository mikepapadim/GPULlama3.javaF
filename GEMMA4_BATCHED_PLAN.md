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
- [ ] batched **NEOX RoPE + per-slot KV write** (own-KV layers) + **RoPE-Q-only** (reuse-KV layers), per-layer freq tables (swa/full)
- [ ] batched **per-head Q/K RMSNorm** (`rmsNormPerHead`) + **V norm** (`rmsNormPerHeadNoWeight`)
- [ ] batched **pre/post RMSNorm** (`applyRmsNorm`) and **norm+residual** (`rmsNormApplyWithResidual`) — B rows
- [ ] batched **GeGLU** gate/up (Q8) — gelu·(up), packed like `batchedFFNSwiGLUFP16Packed` but gelu
- [ ] batched **PLE** tasks: `pleGateGeluMul`, `pleProjScaleAndNormalize`, `addAndScale`, `scaleInPlace`, `scaleInPlaceFromTensor`
- [ ] batched **logit softcap** (`applyLogitSoftcap`) — greedy argmax is softcap-invariant (monotonic), so on-device sampling needs it only for temperature

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

1 / ~15 kernels ported + validated. This is a multi-step port (comparable in size to a full
model backend); progress is committed kernel-by-kernel with a passing microbench each.
