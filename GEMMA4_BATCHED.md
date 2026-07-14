# Gemma 4 on the batched-decode branch — findings

This branch merges **Gemma 4 support (PR #120, `gemma4-new`)** onto the vLLM-style
batched-decode work (`feat/static-batched-decode`, PR #129) and evaluates how much of the
batched-decode / serving stack can be applied to Gemma 4 to squeeze decode performance.

Model: `unsloth/gemma-4-E2B-it-GGUF` → `gemma-4-E2B-it-Q8_0.gguf` (5.0 GB), RTX 4090,
TornadoVM **CUDA backend**, JDK 21.

## Status

- ✅ **Merge is clean** — Gemma 4 (model, loader, tokenizer, kernels, FP16/Q8_0 layers)
  coexists with all batched-decode features; the tree builds and runs.
- ✅ **Stock Gemma 4 GPU decode works** — coherent output:
  `What is the capital of France? → "The capital of France is **Paris**."`
- ✅ **Perf squeeze applied (CUDA graphs, model-agnostic):**

  | Gemma-4-E2B Q8_0 | tok/s | speedup |
  |------------------|------:|--------:|
  | single-token, no CUDA graphs | 22.3 (short) / 11.5 (200-tok) | 1.0× |
  | single-token, **CUDA graphs** | **25.3** / **12.3** | **+13.6% / +6.5%** |

  Enable with `JAVA_TOOL_OPTIONS="-Dllama.cudaGraphs=true"` (or `--cuda-graphs`). Free, output
  unchanged. The gain shrinks with longer context as kernel work grows relative to launch overhead.

## Why full batched decode is a larger follow-up

The batched-decode MMA engine (`BatchedDecodeEngine`) assumes a **uniform** transformer layer
(fixed head/FFN dims, global causal attention, SiLU, adjacent- or split-half RoPE, FP16 weights).
Gemma 4 (a Gemma-3n-class MatFormer) breaks nearly all of these — from `Gemma4Configuration` +
`Gemma4Kernels`:

| Gemma 4 feature | impact on a batched-decode layer graph |
|-----------------|----------------------------------------|
| **Per-layer** head dims (`headDimSwa`/`headDimFull`) + per-layer FFN (`feedForwardLength[]`) | per-layer GEMM shapes — the per-layer graph already supports this, so it's fine |
| **Alternating sliding-window / full attention** (`slidingWindowPattern[]`, `slidingWindowSize`) | needs a **windowed** batched-decode attention kernel (attend `[max(0,pos-W+1), pos]`) |
| **Shared-KV layers** (`sharedKvLayers`) | some layers reuse an earlier layer's KV — cache/block addressing differs per layer |
| **Per-layer embeddings (PLE)** + AltUp/Laurel | embedding path is not a single table lookup |
| **Pre + post norms** around attn and FFN (4/layer) + per-head Q/K RMSNorm + query scaling | extra RMS tasks + Gemma-specific norm kernels |
| **GeGLU** (gelu) not SiLU | fork the packed SwiGLU kernel to GeGLU |
| **NEOX RoPE**, two thetas (`ropeTheta` full / `ropeThetaSwa`) | fork the decode/paged RoPE (NEOX pairing, per-layer theta) |
| **Final logit softcapping** (tanh) | greedy on-device argmax stays valid (softcap is monotonic); temperature needs the softcap applied |
| **BF16 / Q8_0 weights** (no plain FP16) | the MMA path is FP16 (`HalfFloatArray`) — needs a BF16-MMA path or on-load FP16 conversion |

So a Gemma 4 batched-decode layer graph needs, at minimum: a **windowed** decode attention
kernel, a **GeGLU** packed FFN kernel, a **NEOX** decode/paged RoPE kernel, Gemma pre/post-norm
+ Q/K-norm + query-scale tasks, shared-KV addressing, a PLE embedding path, and a BF16-or-FP16
weight story. That is a dedicated multi-file build (comparable to the whole Llama/Qwen3 decode
path), not a small fork like Qwen3 was.

## What already transfers

The **engine-level** serving features are model-agnostic — once a Gemma 4 batched forward
exists, they apply unchanged:

- **On-device sampling** (GPU argmax) — Gemma 4 has a large vocab (~262 k), so the per-step D2H
  logits copy is even bigger than Llama's; on-device argmax would help more. Greedy is valid
  through the tanh softcap (monotonic).
- **Continuous batching**, **PagedAttention**, **prefix caching** — pure scheduling / KV
  addressing, independent of the layer internals (paging must account for shared-KV layers).
- **Logits-skip** on pure-prefill steps.

## Reproduce

```bash
# model
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q8_0.gguf --local-dir .
# stock GPU decode + CUDA graphs
JAVA_TOOL_OPTIONS="-Dllama.cudaGraphs=true" \
  python3 llama-tornado --gpu --cuda --model gemma-4-E2B-it-Q8_0.gguf \
  --prompt "What is the capital of France?" --instruct -n 64
```

## Next

1. Windowed batched-decode attention + GeGLU + NEOX RoPE kernels (the Gemma 4 decode layer graph).
2. FP16 (or BF16-MMA) weight path for the tensor-core GEMMs.
3. Then the model-agnostic engine features (continuous / paging / prefix / on-device sampling)
   drop in — tracked in the roadmap issue (#130).
