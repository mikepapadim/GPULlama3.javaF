# Refactor execution plan — step by step, class by class

The concrete, file-level companion to [`RFC-001-inference-library.md`](RFC-001-inference-library.md)
and [`MODULARIZATION_ROADMAP.md`](MODULARIZATION_ROADMAP.md). Every step names the actual classes to
**add / move / modify / delete** and how to verify. Grounded against `main` @15ee2de.

- Package root: `org.beehive.gpullama3` (abbreviated `~` below).
- Each step is one PR (or a short stack). Merge only on the gates in RFC §6 — chiefly **bit-exact
  logits vs the P0 golden reference** and **no `--bench` regression**.
- Order: `P0 → (M1a ‖ M1b) → M2 → M3 → M4 → M5`, then M6.

---

## P0 — Baseline, golden reference, guardrails

**Land in-flight PRs (order):** `#132 → #128 → #134 → #129`, then `#120`, `#137`. `#131` stays a
findings doc. After #129, `~bench/BatchedDecodeEngine` exists (the M4 source); after #134 the
on-device sampler exists.

**Add**
- `~test/GoldenReference` — dumps per-layer + final logits for `{Llama,Mistral,Qwen2,Qwen3,Phi3,Granite,Devstral} × {FP16,Q8_0}` at a fixed prompt/seed; serializes to `test/golden/*.bin`.
- `~test/BitExactTest` — re-runs each config, asserts byte-identical vs the golden dump.
- `~test/arch/ArchitectureRulesTest` — ArchUnit: `api ⊄ backend`, `model ⊄ uk.ac.manchester.tornado..`, `format ⊄ inference`, with an explicit allowlist of today's violations.

**Verify**: `BitExactTest` green on all 14 configs; `--bench` numbers recorded to `perf-history.jsonl`.

---

## M1a — Library façade + Tensor/GGUF isolation  ·  parallel, low risk

### M1a.1 — Request/result value types
**Add** `~api/`:
- `GenerationRequest` (record + builder): `prompt`, `maxNewTokens`, `temperature`, `topP`, `seed`, `stop`.
- `GenerationResult` (record): `text`, `tokenIds`, `GenerationMetrics`.
- `GenerationToken` (record): `id`, `text`, `logprob`.
- `GenerationMetrics` (record): `promptTokens`, `genTokens`, `prefillTokPerSec`, `decodeTokPerSec`.
- `GenerationListener` (functional): `onToken(GenerationToken)`.
- `InvalidGenerationRequestException`.

**Modify**: none yet. Map `~Options` → `GenerationRequest` in a static helper `GenerationRequests.fromOptions(Options)`.
**Verify**: unit tests for builder validation.

### M1a.2 — Façade over existing engines
**Add** `~api/`:
- `LocalModel` (interface): `GenerationSession newSession(SessionOptions)`, `ModelInfo info()`, `close()`.
- `LocalModels` (final): `static LocalModel load(Path, ModelOptions)` — delegates to `~model/loader/ModelLoader`.
- `GenerationSession` (interface): `GenerationResult generate(GenerationRequest, GenerationListener)`, `close()`.
- `LocalModelImpl`, `GenerationSessionImpl` (package-private) — wrap `~model/Model` + `~inference/InferenceEngine*` + `~tornadovm/plan/ForwardPlanFactory`.
- `ModelOptions`, `SessionOptions` (records + builders): device, execution policy, gpu memory, cuda-graphs.
- `ModelLoadException` (reuse `~model/loader/ModelLoadException`), `ModelExecutionException`, `DeviceMemoryException`, `UnsupportedModelException`.

**Rule**: no `uk.ac.manchester.tornado.*` and no `~tensor/GGUF*` type appears in any `api/` signature. Wrap raw exceptions; preserve cause.
**Verify**: a Java snippet using only `api/` generates text; ArchUnit `api ⊄ backend` passes.

### M1a.3 — Consumers onto the façade
**Modify**:
- `~LlamaApp` — construct `LocalModel`/`GenerationSession` instead of building loaders + engines inline.
- LangChain4j / Quarkus adapters (in the integration repos) — call `api/` only.
**Verify**: CLI output byte-identical for a deterministic prompt (golden-token test).

### M1a.4 — Tensor / GGUF separation
**Move** (package rename, no logic change) `~tensor/ → ~format/gguf/`:
- `GGUF.java`, `GGMLType.java`, `GGMLTensorEntry.java`, `MetadataValueType.java`.
- Keep `~tensor/Float16.java` (numeric helper, not a file format).
**Add** `~tensor/`:
- `Tensor` (interface): `TensorShape shape()`, `DataType dataType()`, `long elementCount()`.
- `TensorShape` (record), `DataType` (enum: `F32,F16,BF16,Q8_0,Q4_0,Q4_K,Q5_K,Q6_K,FP4`).
**Modify**: base classes implement `Tensor` — `~tensor/standard/FloatTensor`, `~tensor/tornado/TornadoTensor` (add the three methods; no copies).
**Verify**: model/op signatures accept `Tensor`, not GGUF/Tornado types; no perf change.

---

## M1b — QuantScheme (operations seam)  ·  keystone

Collapses 36 `~tornadovm/layers/type/{fp16,q8_0}/*` + 14 `~tornadovm/plan/components/{fp16,q8_0}/*`
into **M model templates + D schemes**. Seam detail: [`QUANTSCHEME_SEAM_SCOPING.md`](QUANTSCHEME_SEAM_SCOPING.md).

### M1b.1 — Scheme interface + FP16/Q8_0, one model
**Add** `~tornadovm/layers/scheme/`:
- `QuantScheme` (interface): `DataType dataType()`, `void appendFFN(TaskGraph, LayerCtx)`, `void appendAttention(TaskGraph, LayerCtx)`, `void appendLogits(TaskGraph, LayerCtx)`, `List<Object> transferObjects(LayerCtx)`.
- `FP16Scheme implements QuantScheme` — lifts the kernel calls from `~tornadovm/layers/type/fp16/LlamaFP16FFNLayers` + `LogitsFP16Layer`.
- `Q8_0Scheme implements QuantScheme` — lifts from `~tornadovm/layers/type/q8_0/LlamaQ8_0FFNLayers` (dequant-int8→FP32 path) + Q8 logits.
**Add** `~tornadovm/layers/model/LlamaTransformerLayers` — the single Llama template, generic over `QuantScheme` (replaces the FP16/Q8_0 Llama pair, incl. `decode/` + `prefill/` variants driven by `ExecutionMode`).
**Modify**: `~tornadovm/plan/ForwardPlanFactory` — replace the dtype `if/switch` for Llama with `QuantSchemes.forDataType(dt)`.
**Verify**: Llama-3.2-1B FP16 **and** Q8_0 logits byte-identical vs golden.

### M1b.2 — Migrate the other 6 models
**Add** `~tornadovm/layers/model/{Mistral,Qwen2,Qwen3,Phi3,Granite,Devstral}TransformerLayers` (scheme-driven; per-model hooks for Qwen QK-norm, Granite residual scaling via `ActivationGranite`).
**Delete** after each migration is bit-exact:
- `~tornadovm/layers/type/fp16/*` (18) and `~tornadovm/layers/type/q8_0/*` (18).
**Verify**: all 7 archs × 2 dtypes bit-exact; net LOC down.

### M1b.3 — Collapse plan components + operations framing
**Modify → Delete**:
- `~tornadovm/plan/components/fp16/*PlanComponents` (7) + `~tornadovm/plan/components/q8_0/*PlanComponents` (7) → one scheme-parameterized `~tornadovm/plan/components/{SingleToken,PrefillDecode,BatchPrefillDecode}ForwardPlanComponents` calling `QuantScheme`.
**Add** `~operation/` (the reusable ops the schemes back):
- `QuantizedLinear`, `RmsNorm`, `RotaryEmbedding` (promote `~inference/operation/RoPE`), `Attention`, `FeedForward` — each a descriptor + reference impl; Tornado impls delegate to the scheme.
- `~test/OperationParityTest` — CPU vs Tornado per op, tolerance `|got−ref| ≤ 1e-2·Σ|wᵢaᵢ|+1e-3`.
**Verify**: parity harness green; **anti-explosion test** — a dummy `NoOpScheme` registers and runs with zero central edits.

---

## M2 — Model Architecture SPI

**Add** `~model/arch/`:
- `ModelArchitecture` (interface): `boolean matches(GgufMetadata)`, `Model createModel(ModelSource)`, `Tokenizer tokenizer()`, `ChatFormat chatFormat()`, `TransformerLayers layers()`.
- `ModelArchitectures` (final, static registry): `register(ModelArchitecture)`, `ModelArchitecture match(GgufMetadata)`, `List<String> ids()`.
- One provider per family: `LlamaArchitecture`, `MistralArchitecture`, `Qwen2Architecture`, `Qwen3Architecture`, `Phi3Architecture`, `GraniteArchitecture`, `DevstralArchitecture` — each composes its existing `~model/<fam>/*`, `~model/loader/<Fam>ModelLoader`, `~tokenizer/<Fam>Tokenizer`, `~model/format/<Fam>ChatFormat`, `~tornadovm/layers/model/<Fam>TransformerLayers`.

**Modify → Delete**:
- `~model/loader/ModelLoader.detectModelType(...)` switch → `ModelArchitectures.match(md)`.
- `~model/ModelType` — demote to a metadata key (or delete once providers own detection).
- `~inference/InferenceEngine.generateTokens{Llama,Qwen3,…}` statics — removed; the engine drives `ModelArchitecture.layers()`.

**Verify**: all archs load via the registry, bit-exact; a throwaway `DummyArch` registers + runs with no central edit.

---

## M3 — Executor / Backend seam (+ attention-backend seam)

**Add** `~runtime/exec/`:
- `ForwardExecutor` (interface): `ingestPrefill(State, int[] tokens)`, `float[] decodeStep(State, int token)`, `float[] logits(State)`.
- `TornadoForwardExecutor` — wraps `~tornadovm/TornadoVMMasterPlan{SingleToken,PrefillDecode,BatchPrefillDecode}` + `~tornadovm/plan/ForwardPlan*` as `CompiledProgram`s.
- `CpuForwardExecutor` — wraps the CPU path in `~inference/InferenceCore*`.

**Add** `~runtime/backend/`:
- `Backend` (interface): `String id()`, `List<Device> devices()`, `DeviceCapabilities capabilities(Device)`, `ProgramCompiler compiler(Device)`.
- `Device`, `DeviceSelector` (`preferredGpu()`, `cpuFallback()`, `requireCapability(...)`), `DeviceCapabilities` (from `~tornadovm/TensorCoreSupport` — MMA/tensor-core, dtypes).
- `ProgramCompiler`, `CompiledProgram` (wrap `ForwardPlanFactory` output; cache key = arch × dtype × mode × shape).
- `TornadoBackend`, `CpuBackend`.

**Add** attention-backend seam (vLLM-derived, for a future flash-attention kernel):
- `~operation/AttentionOp` (descriptor) + `~runtime/backend/AttentionBackend` (interface); `TornadoAttentionBackend` default.

**Modify → Delete**:
- `~model/Model.run{Interactive,InstructOnce,InstructOnceLangChain4J}` + every inline `if(useTornadovm)` → one engine loop selecting a `ForwardExecutor` by capability.
- `~tornadovm/TensorCoreSupport` → folded into `DeviceCapabilities`.

**Verify**: CPU and Tornado paths bit-exact; capability query drives MMA gating; CLI + LangChain4j unchanged.

---

## M4 — LLMEngine core  ·  keystone

Promote `~bench/BatchedDecodeEngine` (lands in P0 via #129) into a reusable core.

**Add** `~runtime/engine/`:
- `LLMEngine`: `RequestHandle addRequest(GenerationRequest, GenerationListener)`, `void step()`, `boolean hasWork()`.
- `EngineCore` (sync loop) + `AsyncLLMEngine` (frontend) — the vLLM v1 split.
- `Scheduler` (interface) → `StaticScheduler`, `ContinuousScheduler` (extract `runContinuous`); **chunked prefill** — mix prefill+decode tokens per step.
- `KVCacheManager` (interface) → `ContiguousKVCache`, `PagedKVCache`; `KVCacheSpec` (layout decoupled from model); `BlockPool`, `BlockTable`.
- `PrefixCache` (hash prefix → shared blocks).
- `Sampler` strategy — wire `~inference/sampler/{Sampler,CategoricalSampler,ToppSampler}` + the on-device sampler (#134) behind one interface; add a `LogitsProcessor` chain.

**Modify → façade**: `~bench/BatchedDecodeEngine` becomes a thin caller of `LLMEngine`; its `-Dbatch.decode.*` props map to `SessionOptions`.

**Verify** (vs `BATCHED_DECODE.md` bench parity): paged ~10.7× less KV; continuous +20% throughput / +24% util; prefix +85% on shared-prefix batches; one loaded model → two independent sessions with independent position/KV/sampler state.

---

## M5 — Server on the engine

**Modify**:
- `~server/InferenceService` → thin adapter: request → `LLMEngine.addRequest`; a server loop calls `step()` and streams per-request deltas.
- `~server/OpenAIServer` — unchanged above the service; N concurrent clients share the batched engine.

**Verify**: `/v1/chat/completions` streams correctly under 1 client; throughput scales with client count; single-client latency not regressed; prefix-cache hit-rate logged.

---

## M6 — Payoff & seams (parallel, ongoing)

**Add** (new `QuantScheme`s — the whole point of M1b):
- `~tornadovm/layers/scheme/Q4KScheme`, `FP4Scheme` + `~tensor/tornado/{Q4_KTornadoTensor,FP4TornadoTensor}` + an activation-quant kernel + IMMA/FP4 matmul kernels. Correctness spec: jam `PRECISION.md`.
- Quantized-KV variant of `KVCacheManager`; `SpecDecode` hook (draft model / EAGLE) on `EngineCore`.
- Flash-attention kernel behind `AttentionBackend`.
**Add** (design-only): `~runtime/distributed/{ShardPlan,ParallelExecutor}` interfaces.
**Add** (providers): Gemma 4 (`#120`) as `GemmaArchitecture`; MoE / SSM providers.

**Verify**: FP4 GEMM parity on sm_120 (or clean UNSUPPORTED gating pre-hardware); unlocks the RTX 5090 FP4/FP8/INT8 tensor cores currently unused.

---

## Appendix — current → target class map

| Current (`main`) | Target |
|---|---|
| `Options`, `LlamaApp` | `api/GenerationRequest` + `api/{LocalModel,GenerationSession}`; `LlamaApp` = consumer |
| `tensor/{GGUF,GGMLType,GGMLTensorEntry,MetadataValueType}` | `format/gguf/*` |
| `tensor/{standard,tornado}/*` | implement `tensor/Tensor` |
| `tornadovm/layers/type/{fp16,q8_0}/*` (36) | `tornadovm/layers/scheme/{FP16,Q8_0}Scheme` + `layers/model/<Fam>TransformerLayers` |
| `tornadovm/plan/components/{fp16,q8_0}/*` (14) | scheme-parameterized components |
| `inference/operation/RoPE` | `operation/RotaryEmbedding` (+ `RmsNorm`, `QuantizedLinear`, `Attention`, `FeedForward`) |
| `model/loader/ModelLoader.detectModelType` + `model/ModelType` | `model/arch/ModelArchitectures` registry + `<Fam>Architecture` providers |
| `inference/InferenceEngine{,WithPrefillDecode,WithBatchPrefillDecode}` + `InferenceCore*` | `runtime/exec/ForwardExecutor` (`Tornado`/`Cpu`) + `runtime/engine/LLMEngine` |
| `inference/state/{Llama,Qwen2,Qwen3,Phi3,Granite,Devstral}State` | common `runtime/state/TransformerState` + `KvCache`; model-specific only where it truly differs |
| `tornadovm/TornadoVMMasterPlan*` + `plan/ForwardPlan*` | `runtime/backend/CompiledProgram` behind `ProgramCompiler` |
| `tornadovm/TensorCoreSupport` | `runtime/backend/DeviceCapabilities` |
| `bench/BatchedDecodeEngine` (arrives via #129) | `runtime/engine/{LLMEngine,Scheduler,KVCacheManager,BlockPool,PrefixCache}` |
| `server/InferenceService` | thin adapter over `LLMEngine` |
