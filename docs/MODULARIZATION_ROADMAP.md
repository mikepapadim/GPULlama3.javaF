# GPULlama3 — Modularization Roadmap & Implementation Plan

Turns the approved 6-layer design into sequenced, PR-sized work. Every phase is
behavior-preserving and merges only on **bit-exact logits + no throughput regression**.

- **Design reference**: 6 layers — ModelArch SPI (1), QuantScheme (2), ForwardExecutor (3),
  LLMEngine core (4), Distributed seam (5), Server-on-engine (6).
- **Anchor issue**: #130 (vLLM-style serving roadmap). This roadmap is the *architecture* track
  that generalizes what #129 proved.
- **Prime directive**: promote / generalize / de-duplicate. Do **not** re-invent features that
  already exist in `bench/BatchedDecodeEngine`.

### Reconciliation — folded-in library-API design (2026-07-22)

This roadmap is the spine. A second, library-API-first design (public façade for three audiences —
app developer, advanced inference developer, backend contributor) is **merged in** as additive
layers, not adopted wholesale:

| Library-API track | Where it lands here | Verdict |
|---|---|---|
| Public API façade (`LocalModel`/`GenerationSession`/`GenerationRequest`/`GenerationResult`) | **new M1a** | ADOPT — strangler entry; protects LangChain4j/Quarkus |
| Tensor/GGUF separation (generic `Tensor` + `format/gguf`) | **M1a** | ADOPT — low-risk API-boundary cleanup |
| Operations layer (RMSNorm/RoPE/**QuantizedLinear**/Attention/FFN + reference+Tornado + parity) | **M1b** (QuantScheme) | MERGE — ops layer *is* the QuantScheme + low-bit-matmul seam |
| Program / CompiledProgram split | M3 | same idea — evolve `tornadovm/plan/*` |
| Backend / Device SPI | M3 | same idea — from `TensorCoreSupport` |
| Model provider SPI | M2 | same idea — static registry (D2) |
| Loaded-model vs session state / `KvCache` | M4 | fold into LLMEngine promotion |
| Maven modules (15) | deferred | packages first, modules later |

**Framing kept from the library design**: three audiences (generation/sessions → operations/tensors
→ backend/kernels), a strangler migration (façade in front of today's engines), ArchUnit dependency
enforcement, and a CPU/Tornado operation parity harness. **Corrections applied to it**: it omitted
that the vLLM serving features are trapped in `bench/BatchedDecodeEngine` (→ M4), that the 32-class
dtype explosion is the load-bearing blocker (→ M1b first), and that the whole point is enabling
low-bit GPU matmul (Q4→FP4) — not cleanliness. **First move (locked)** = strangler: **M1a façade +
M1b QuantScheme together**.

---

## 0. Pre-flight — land in-flight work first (decision D1)

The refactor must branch from a settled base. In-flight PRs touch the exact files being
refactored; rebasing them onto the refactor is more painful than the reverse.

| PR | Why it blocks | Action |
|----|---------------|--------|
| #129 static/continuous batched decode | Source of the LLMEngine internals (layer 4) | **Land first** |
| #134 on-device sampling | Becomes the `Sampler` strategy (layer 4) | Land first |
| #120 Gemma 4 | New arch; easier as a *provider* after phase 2 than merged before | Land, then migrate in phase 7 |
| #132 Qwen3 rmsnorm race fix | Correctness baseline for bit-exact gate | Land first |

**Exit gate P0**: `master` contains #129/#134/#132; `--bench` + `BATCHED_DECODE.md` verifies
pass; captured as the **golden reference** (per-layer + final-logit dumps, per arch × dtype).

---

## Milestone map

```
M0 Baseline ──► M1a Façade + Tensor/GGUF ──► M1b QuantScheme ──► M2 ModelArch SPI ──► M3 Executor seam
   (P0)           (library shell)              (layer 2)           (layer 1)            (layer 3)
                                                                                            │
                                                                                            ▼
                                        M4 LLMEngine core ──► M5 Server-on-engine ──► M6 Seams + payoff
                                           (layer 4)            (layer 6)               (layers 5,7)
```

Critical path: M1b → M2 → M3 → M4 → M5. M6 rides on top. M1a runs in parallel (independent, low
risk). M1b and the golden reference (P0) are the only true prerequisites for everything else.

---

## M1a — Library façade + Tensor/GGUF isolation  ·  library shell  ·  ~3–4 PRs

**Goal**: make the project *look and embed like a library* up front, cheaply, with zero kernel
changes — a strangler façade in front of today's engines. Independent of the critical path.

| PR | Scope | Acceptance |
|----|-------|-----------|
| M1a.1 | `api/` — `GenerationRequest`/`GenerationResult`/`GenerationToken` (immutable, validated); map current `Options` into it | unit tests; no execution change |
| M1a.2 | `LocalModel`/`LocalModels.load`/`GenerationSession` façade delegating to existing loaders + `InferenceEngine*` + `ForwardPlanFactory` | README Java example uses only `api/`; **no TornadoVM or GGUF type crosses the façade** |
| M1a.3 | `LlamaApp` + LangChain4j/Quarkus integrations become façade consumers | CLI output byte-identical for a deterministic prompt; consumers compile unchanged |
| M1a.4 | Move GGUF types out of `tensor/` → `format/gguf/` (`GGUF`, `GGMLType`, `GGMLTensorEntry`, `MetadataValueType`); introduce generic `Tensor`/`TensorShape`/`DataType` that current `tensor/standard/*` + `tensor/tornado/*` implement | no copies, no perf change; model/op APIs accept `Tensor`, not GGUF/Tornado types |

**Files**: new `api/*`, new `format/gguf/*` (move from `tensor/`), new `tensor/{Tensor,TensorShape,DataType}.java`,
`LlamaApp.java` (thin to a consumer), `integration/langchain4j|quarkus` adapters.
**Public-exception hygiene**: no raw TornadoVM/GGUF exception crosses `api/` (wrap, preserve cause).
**Why first**: this is where "looks like a library" is won — and it is decoupled from the
load-bearing internal work, so it can proceed in parallel with M1b.

---

## M1b — QuantScheme extraction (operations seam)  ·  layer 2  ·  ~2–3 PRs

**Goal**: dtype becomes data. Collapse 32 `layers/type/{fp16,q8_0}` classes to M model-templates
+ D schemes. This is also the **operations layer**: express `QuantizedLinear` (+ RMSNorm/RoPE/
Attention/FFN) as reusable operations with a reference impl and a **CPU/Tornado parity harness**,
so a Java dev can call an op without a full model — and so Q4/FP4 land as a new *scheme*, not a new
class-tree. Detailed seam design already in `gpullama3-dtype-seam-scoping.md`.

| PR | Scope | Acceptance |
|----|-------|-----------|
| M1.1 | `QuantScheme` interface + `FP16Scheme`, `Q8_0Scheme`; make `LlamaFFNLayers` generic over it | Llama-3.2-1B F16 & Q8_0 logits **byte-identical** vs golden |
| M1.2 | Migrate remaining 6 models to generic template + model hooks; delete `type/{fp16,q8_0}` dups | All 7 archs bit-exact; net LOC down |
| M1.3 | Merge `plan/components/{fp16,q8_0}` → scheme-parameterized; factory dtype switch → `schemeFor()` | Same |

**Files**: `tornadovm/layers/AbstractTransformerLayerTaskGraphs.java`, new
`tornadovm/layers/scheme/*`, `tornadovm/plan/ForwardPlanFactory.java`, `tensor/tornado/*`.
**Risk**: TornadoVM buffer-lifetime / `consumeFromDevice` predecessor-name subtlety — keep
transfer lists inside the scheme. **Payoff proof**: adding a dummy scheme needs zero central edits.

---

## M2 — Model Architecture SPI  ·  layer 1  ·  ~3 PRs

**Goal**: one provider per arch; adding a model touches no central file.

| PR | Scope | Acceptance |
|----|-------|-----------|
| M2.1 | `ModelArchitecture` interface + `ModelArchitectures` registry; port Llama behind it; `matches(md)` replaces `detectModelType` | Llama loads + runs identical; `detectModelType` Llama branch deleted |
| M2.2 | Migrate the other 6 archs; delete per-model `*ModelLoader` + `detectModelType` chain | All archs load via registry; bit-exact |
| M2.3 | Remove `InferenceEngine.generateTokens{Llama,Qwen3,…}` statics — engine drives arch layer graphs | Single generate path |

**Files**: new `model/arch/*` providers, `model/loader/*` (collapse), `model/loader/ModelLoader.detectModelType` (delete), `inference/InferenceEngine.java`.
**Decision D2**: `ServiceLoader` vs static registry → **static registry** (no native-image config).
**Payoff proof**: a throwaway dummy arch registers + runs with no central edit.

---

## M3 — ForwardExecutor seam  ·  layer 3  ·  ~2 PRs

**Goal**: CPU and TornadoVM behind one interface; the three `Model` generate loops collapse.

| PR | Scope | Acceptance |
|----|-------|-----------|
| M3.1 | `ForwardExecutor` (`ingestPrefill`/`decodeStep`/`logits`) + `CpuForwardExecutor`, `TornadoForwardExecutor`; `TensorCoreSupport` → `DeviceCapabilities` | Both paths bit-exact; capability query drives MMA gating |
| M3.2 | Collapse `Model.run{Interactive,InstructOnce,InstructOnceLangChain4J}` + all `if(useTornadovm)` into one engine loop | CLI + LangChain4j behavior unchanged |

**Files**: new `runtime/exec/*`, `inference/InferenceCore*` (wrap), `tornadovm/TensorCoreSupport.java` → `DeviceCapabilities`, `model/Model.java` (thin the defaults).
**Note**: reserves a future jam-style Vector-API/int8 CPU executor with zero engine changes.

---

## M4 — LLMEngine core  ·  layer 4  ·  ~4 PRs  (the keystone)

**Goal**: promote `bench/BatchedDecodeEngine` internals into a reusable, model/dtype/backend-
agnostic core. **Decision D3**: package home → `runtime/engine/` (bench becomes a thin caller).

| PR | Scope | Acceptance |
|----|-------|-----------|
| M4.1 | Extract `KVCacheManager` (`Contiguous` + `Paged`) + `BlockPool` + `BlockTable` from the property-driven code | Paged path matches bench numbers (~10.7× less KV, ~1% overhead) |
| M4.2 | `Scheduler` (`Static` + `Continuous`); `runContinuous` → `ContinuousScheduler` | Continuous: +20% throughput / +24% util vs static (bench parity) |
| M4.3 | `PrefixCache` (hash prefix → shared blocks) | +85% throughput / 85% fewer prefill tokens on shared-prefix batch |
| M4.4 | `LLMEngine.addRequest()/step()` tying it together over ModelArch graphs + ForwardExecutor + Sampler; bench `main` becomes a caller | `bench/BatchedDecodeEngine` verifies still pass through the new core |

**Files**: `runtime/engine/{LLMEngine,Scheduler,KVCacheManager,BlockPool,PrefixCache}.java`,
`bench/BatchedDecodeEngine.java` (gut to a façade), `inference/sampler/*` (wire as strategy).
**Risk**: highest. Gate each PR against the exact `BATCHED_DECODE.md` reproductions.

---

## M5 — Server on the engine  ·  layer 6  ·  ~2 PRs

**Goal**: the OpenAI server actually batches concurrent requests.

| PR | Scope | Acceptance |
|----|-------|-----------|
| M5.1 | `InferenceService` → thin adapter: request → `LLMEngine.addRequest`; server loop calls `step()`, streams per-request deltas | `/v1/chat/completions` streams correctly under 1 client |
| M5.2 | Concurrency: N simultaneous clients share the batched engine; prefix-cache hit-rate logged | Throughput scales with concurrency; single-client latency not regressed |

**Files**: `server/InferenceService.java`, `server/OpenAIServer.java` (unchanged above service).
**This is the headline outcome**: "batching in a benchmark" → "the server batches."

---

## M6 — Seams + payoff  ·  layers 5 & 7  ·  ongoing

| Track | Scope | Note |
|-------|-------|------|
| Distributed seam (layer 5) | `ShardPlan` + `ParallelExecutor` interfaces; device index through KV/weights/executor | **Design only** until TornadoVM multi-device |
| Q4_K / FP4 schemes (layer 2) | New `QuantScheme` + tensors; needs the low-bit MMA track below | Closes qxotic quant-breadth gap |
| New archs (layer 1) | Gemma 4 (#120), then MoE / SSM providers | Closes qxotic model-breadth gap |
| Prefix→prompt cache | Extend `PrefixCache` to persisted session resume (qxotic-style) | Closes the 84–404× resume gap |

### Parallel track — native low-bit / FP4 GPU compute (spans TornadoVM)
Independent of M1–M5 but rides the QuantScheme seam. From the earlier flagship analysis:
1. TornadoVM: int8 (`m16n8k32.s8`) + FP8/FP4 MMA intrinsics + `dp4a` fallback (extends jTile work).
2. GPULlama3: activation-quant kernel + IMMA matmul kernels + `FP4Scheme`.
Correctness spec = jam `PRECISION.md` (`|got−ref| ≤ 1e-2·Σ|wᵢaᵢ|+1e-3`, `n∈{7,8,13,16}` sweep).
Payoff: unlocks RTX 5090 sm_120 FP4 tensor cores currently unused.

---

## Global acceptance gates (every PR)

1. **Bit-exact**: per-layer + final-logit equality vs the M0 golden reference, per arch × dtype
   (deterministic greedy).
2. **No regression**: `--bench` (#133) tok/s within noise; `perf-history.jsonl` (#113/#114) tracks delta.
   Baseline best config = `--cuda-graphs --with-prefill-decode --batch-prefill-size ≥ promptLen`.
3. **Anti-explosion invariant**: for M1b/M2, a dummy scheme/arch registers and runs with **zero**
   edits to central files (enforced by a unit test).
4. **Serving verifies**: M4/M5 pass the `BATCHED_DECODE.md` reproductions verbatim.
5. **ArchUnit dependency rules** (added P0, allowlist shrinks each milestone): `api` ⊄ backend,
   `model` ⊄ `uk.ac.manchester.tornado..`, `format` ⊄ inference engine, `tokenizer` ⊄ backend.
6. **Operation parity harness** (M1b onward): CPU vs Tornado per op, tolerance
   `|got−ref| ≤ 1e-2·Σ|wᵢaᵢ| + 1e-3` (jam `PRECISION.md`); skips cleanly with no GPU.

## Success metrics
- Adding a dtype: **~23 files → 1 scheme + 1 tensor + 1 line.**
- Adding a model: **~10 packages → 1 provider.**
- Server: single-stream → **concurrent batched** (throughput scales with client count).
- Feature reach: vLLM mechanics generalized past **LLaMA/Qwen3/FP16/CUDA**.

## Open decisions (blockers to start)
- **D1** sequencing vs #129/#120/#134 — recommend land-first. *(gates M0)*
- **D2** SPI mechanism — recommend static registry. *(gates M2)*
- **D3** engine package — recommend `runtime/engine/`, bench façade. *(gates M4)*
- **D4** distributed depth — recommend seams-only this round. *(gates M6)*

## Suggested ordering
`P0 → (M1a ‖ M1b) → M2 → M3 → M4 → M5`, then M6 tracks in parallel. **M1a** (façade + tensor/GGUF)
and **M1b** (QuantScheme) run in parallel: M1a is the cheap, visible "looks-like-a-library" shell
and is independent of the critical path; **M1b is the load-bearing first move** — mechanical,
low-risk, and it unblocks both the dtype-breadth gap (Q4/FP4) and every later phase.
