package org.beehive.gpullama3.bench;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.kernels.Gemma4BatchDecodeKernels;
import org.beehive.gpullama3.tornadovm.kernels.Gemma4Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.ArrayList;
import java.util.List;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/**
 * Batched-decode engine for Gemma 4 (Q8_0). B independent sequences, greedy, one token/step,
 * each with its own KV region. Mirrors the single-token {@code Gemma4Q8_0FFNLayers} task order
 * batched: Q8 tensor-core GEMMs for the projections + the validated batched Gemma kernels
 * (windowed attention, NEOX RoPE, per-head norms, GeGLU, sandwich norms, PLE). Correctness:
 * B copies of one prompt (greedy) → all streams must be identical AND coherent.
 */
public class Gemma4BatchedDecodeEngine {

    static final int RMS_LOCAL = 256, HEAD_NORM_LOCAL = 64;
    final KernelContext ctx = new KernelContext();

    // config/dims
    Gemma4Configuration config;
    Gemma4TornadoWeights w;
    int B, paddedB, dim, vocab, nLayers, nHeads, nHeadKv, kvMul, nEmbdPerLayer, perLayerTotal, decodeCtx;
    int maxHeadDim, maxFFN, maxQDim, maxKvDim, slotStride;
    int[] cacheBaseOff;
    float eps;

    // buffers
    FloatArray wrapX, qB, kB, vB, woOut, gateUp, w2Out, plInputs, plScratch, plGate, plOut, plTokRow, keyCache, valCache, logits, plModelProjF32;
    // distinct RMS-scale buffers per reduce→apply pair (reusing one buffer races on the GPU).
    FloatArray scAtt, scPAtt, scFfn, scPFfn, scPle, scLog;
    HalfFloatArray normed, normedFfn, attnOut, hb, normedFinal;
    IntArray seqPos, sampled;

    public static void main(String[] args) throws Exception {
        new Gemma4BatchedDecodeEngine().run(args);
    }

    void run(String[] args) throws Exception {
        B = Integer.getInteger("gemma.B", 8);
        decodeCtx = Integer.getInteger("gemma.ctx", 256);
        int nDecode = Integer.getInteger("gemma.n", 32);
        boolean cudaGraphs = Boolean.parseBoolean(System.getProperty("gemma.cudaGraphs", "false"));
        System.setProperty("llama.prefillBatchSize", String.valueOf(B));

        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        config = (Gemma4Configuration) model.configuration();
        w = (Gemma4TornadoWeights) model.weights();
        dim = config.dim(); vocab = config.vocabularySize(); nLayers = config.numberOfLayers();
        nHeads = config.numberOfHeads(); nHeadKv = config.numberOfKeyValueHeads(); kvMul = config.kvMul();
        nEmbdPerLayer = config.embeddingLengthPerLayer(); perLayerTotal = nLayers * nEmbdPerLayer;
        eps = config.rmsNormEps();
        maxHeadDim = config.maxHeadDim(); maxFFN = config.maxFeedForwardLength();
        maxQDim = nHeads * maxHeadDim; maxKvDim = nHeadKv * maxHeadDim;
        paddedB = (B + 127) & ~127;

        // Per-slot KV cache offsets (capped at decodeCtx).
        cacheBaseOff = new int[nLayers];
        int running = 0;
        for (int l = 0; l < nLayers; l++) {
            int reuse = config.kvReuseLayer(l);
            if (reuse < 0) { cacheBaseOff[l] = running; running += decodeCtx * (nHeadKv * config.headDim(l)); }
            else cacheBaseOff[l] = cacheBaseOff[reuse];
        }
        slotStride = Math.max(1, running);

        ChatFormat cf = model.chatFormat();
        List<Integer> prompt = new ArrayList<>();
        if (model.shouldAddBeginOfText()) prompt.add(cf.getBeginOfText());
        prompt.addAll(cf.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        int P = prompt.size();
        var stopTokens = cf.getStopTokens();
        System.out.printf("[gemma] B=%d ctx=%d P=%d n=%d dim=%d vocab=%d layers=%d slotStride=%d%n",
                B, decodeCtx, P, nDecode, dim, vocab, nLayers, slotStride);

        if (Boolean.getBoolean("gemma.cpuRef")) {
            try {
                Options cpu = new Options(options.modelPath(), options.prompt(), options.systemPrompt(), options.suffix(),
                        false, options.temperature(), options.topp(), options.seed(), options.maxTokens(), false, false,
                        false, false, 1);
                Model cpuModel = loadModel(cpu);
                var cpuState = cpuModel.createNewState();
                int pp = 0, first = -1;
                for (int t : prompt) { cpuModel.forward(cpuState, t, pp++); }
                int am = 0; float best = -1e30f;
                for (int i = 0; i < vocab; i++) { float v = cpuState.logits.getFloat(i); if (v > best) { best = v; am = i; } }
                System.out.printf("[cpuref] argmax after prompt = %d  ('%s')%n", am, model.tokenizer().decode(List.of(am)));
            } catch (Throwable e) { System.out.println("[cpuref] failed: " + e); }
        }

        allocate();
        GridScheduler gs = new GridScheduler();
        List<ImmutableTaskGraph> graphs = new ArrayList<>();
        for (int l = 0; l < nLayers; l++) graphs.add(buildLayer(l, gs).snapshot());
        graphs.add(buildLogits(gs).snapshot());
        int logitsIdx = nLayers;

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graphs.toArray(new ImmutableTaskGraph[0]))) {
            int[][] streams = new int[B][nDecode];
            int[] cur = new int[B];
            long t0 = System.nanoTime();
            // prompt prefill (logits ignored except last) + decode
            for (int step = 0; step < P + nDecode; step++) {
                boolean prefill = step < P;
                int pos = step;
                for (int b = 0; b < B; b++) {
                    int tok = prefill ? prompt.get(step) : cur[b];
                    loadEmbedRow(b, tok);
                    seqPos.set(b, pos);
                }
                gatherPLE(prefill ? new int[]{prompt.get(step)} : cur, prefill);
                for (int l = 0; l < nLayers; l++) {
                    execGraph(plan, gs, l, cudaGraphs);
                    if (step == 0 && Boolean.getBoolean("gemma.dbgL0") && (l == 0 || l == 13 || l == 14 || l == 15 || l == 25 || l == nLayers - 1)) {
                        double mn = 1e30, mx = -1e30, ss = 0; for (int i = 0; i < dim; i++) { float v = wrapX.get(i); mn = Math.min(mn, v); mx = Math.max(mx, v); ss += v * v; }
                        System.out.printf("[dbgL] L%d wrapX: min=%.3f max=%.3f rms=%.4f  (own-KV=%b)%n", l, mn, mx, Math.sqrt(ss / dim), config.hasOwnKv(l));
                    }
                }
                if (!prefill || step == P - 1) {
                    execGraph(plan, gs, logitsIdx, cudaGraphs);
                    if (step == P - 1) {
                        double mn = 1e30, mx = -1e30, sum = 0; int nan = 0;
                        for (int i = 0; i < vocab; i++) { float v = logits.get(i); if (Float.isNaN(v)) nan++; else { mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; } }
                        System.out.printf("[dbg] logits slot0: min=%.3f max=%.3f mean=%.3f nan=%d argmax=%d%n", mn, mx, sum / vocab, nan, sampled.get(0));
                        double wmn = 1e30, wmx = -1e30, wsum = 0; for (int i = 0; i < dim; i++) { float v = wrapX.get(i); wmn = Math.min(wmn, v); wmx = Math.max(wmx, v); wsum += v; }
                        System.out.printf("[dbg] wrapX(final) min=%.3f max=%.3f mean=%.4f | embedType=%s%n", wmn, wmx, wsum / dim, w.getTokenEmbeddingTable().type());
                    }
                }
                if (step >= P - 1) {
                    int s = step - (P - 1);
                    if (s < nDecode) for (int b = 0; b < B; b++) { cur[b] = sampled.get(b); streams[b][s] = cur[b]; }
                }
            }
            long ns = System.nanoTime() - t0;

            boolean identical = true;
            for (int b = 1; b < B; b++) for (int s = 0; s < nDecode; s++) if (streams[b][s] != streams[0][s]) identical = false;
            StringBuilder txt = new StringBuilder();
            for (int s = 0; s < nDecode; s++) { int tk = streams[0][s]; if (stopTokens.contains(tk)) break; txt.append(model.tokenizer().decode(List.of(tk))); }
            System.out.print("[dbg] slot0 toks:"); for (int s = 0; s < Math.min(6, nDecode); s++) System.out.print(" " + streams[0][s]); System.out.println();
            if (B > 1) { System.out.print("[dbg] slot1 toks:"); for (int s = 0; s < Math.min(6, nDecode); s++) System.out.print(" " + streams[1][s]); System.out.println(); }
            System.out.println("\n──────── slot 0 ────────\n" + txt + "\n────────────────────────");
            System.out.printf("[verify] all %d streams identical: %b%n", B, identical);
            System.out.printf("[perf] %d steps, %.1f ms total, %.1f ms/step%n", P + nDecode, ns / 1e6, ns / 1e6 / (P + nDecode));
        }
    }

    void allocate() {
        wrapX = f(B * dim); normed = h(paddedB * dim); normedFfn = h(paddedB * dim);
        scAtt = f(B); scPAtt = f(B); scFfn = f(B); scPFfn = f(B); scPle = f(B); scLog = f(B);
        qB = f(paddedB * maxQDim); kB = f(paddedB * maxKvDim); vB = f(paddedB * maxKvDim);
        attnOut = h(paddedB * maxQDim); woOut = f(paddedB * dim);
        gateUp = f(paddedB * 2 * maxFFN); hb = h(paddedB * maxFFN); w2Out = f(paddedB * dim);
        plInputs = f(B * perLayerTotal); plScratch = f(B * perLayerTotal);
        plGate = f(B * nEmbdPerLayer); plOut = f(B * dim); plTokRow = f(B * perLayerTotal);
        keyCache = f(B * slotStride); valCache = f(B * slotStride); keyCache.init(0f); valCache.init(0f);
        tmpRow = f(perLayerTotal);
        // perLayerModelProj is F16 — dequant to F32 host-side (kernels can't read HalfFloatArray.get()).
        var src = w.perLayerModelProj.asHalfFloatArray();
        plModelProjF32 = new FloatArray(perLayerTotal * dim);
        for (int i = 0; i < perLayerTotal * dim; i++) plModelProjF32.set(i, src.get(i).getFloat32());
        seqPos = new IntArray(B);
        normedFinal = h(paddedB * dim); logits = f(paddedB * vocab); sampled = new IntArray(paddedB);
    }

    static FloatArray f(int n) { return new FloatArray(n); }
    static HalfFloatArray h(int n) { HalfFloatArray a = new HalfFloatArray(n); a.init(new HalfFloat(0f)); return a; }

    // ── host embedding + PLE gather ──────────────────────────────────────────
    void loadEmbedRow(int b, int token) {
        // raw embedding; the sqrt(dim) scale is applied on-device by the layer-0 _embscale task.
        var t = w.getTokenEmbeddingTable();
        switch (t.type()) {
            case F32 -> { var a = t.asFloatArray(); for (int i = 0; i < dim; i++) wrapX.set(b * dim + i, a.get(token * dim + i)); }
            case F16 -> { var a = t.asHalfFloatArray(); for (int i = 0; i < dim; i++) wrapX.set(b * dim + i, a.get(token * dim + i).getFloat32()); }
            case Q8_0 -> { var a = t.asByteArray(); int bpr = dim / 32; for (int j = 0; j < dim; j++) { int blk = (token * bpr + j / 32) * 34; float s = a.getHalfFloat(blk).getFloat32(); wrapX.set(b * dim + j, a.get(blk + 2 + j % 32) * s); } }
            default -> throw new UnsupportedOperationException("embed " + t.type());
        }
    }

    void gatherPLE(int[] tokens, boolean allSame) {
        float sc = (float) Math.sqrt(nEmbdPerLayer);
        for (int b = 0; b < B; b++) {
            int token = allSame ? tokens[0] : tokens[b];
            ModelLoader.copyEmbeddingRowToFloatArray(w.perLayerTokenEmbd, token, perLayerTotal, tmpRow, 1.0f);
            for (int i = 0; i < perLayerTotal; i++) plTokRow.set(b * perLayerTotal + i, tmpRow.get(i) * sc);
        }
    }
    FloatArray tmpRow;

    // ── graph builders ───────────────────────────────────────────────────────
    TaskGraph buildLayer(int l, GridScheduler gs) {
        int headDim = config.headDim(l), qDim = nHeads * headDim, kvDim = nHeadKv * headDim, ffn = config.feedForwardLength(l);
        boolean own = config.hasOwnKv(l), swa = config.isSwa(l);
        int window = swa ? config.slidingWindowSize() : decodeCtx;
        FloatArray fcr = (swa ? w.freqCisRealSwa : w.freqCisRealFull).asFloatArray();
        FloatArray fci = (swa ? w.freqCisImagSwa : w.freqCisImagFull).asFloatArray();
        int peOff = l * nEmbdPerLayer, base = cacheBaseOff[l];
        String name = "L" + l;
        TaskGraph g = new TaskGraph(name);

        if (l == 0) {
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, wrapX, seqPos, plTokRow);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, ctx, normed, normedFfn, scAtt, scPAtt, scFfn, scPFfn, scPle, qB, kB, vB, attnOut, woOut, gateUp, hb, w2Out,
                    keyCache, valCache, plInputs, plScratch, plGate, plOut, plModelProjF32, w.perLayerProjNorm.asFloatArray());
            // PLE setup (layer 0 only)
            g.task(name + "_embscale", Gemma4Kernels::scaleInPlace, ctx, wrapX, (float) Math.sqrt(dim), B * dim);
            g.task(name + "_plmodel", Gemma4BatchDecodeKernels::batchedMatVecF32, ctx, wrapX, plModelProjF32, plScratch, dim, perLayerTotal);
            g.task(name + "_plnorm", Gemma4BatchDecodeKernels::batchedGemmaPleProjScaleAndNormalize, ctx, plScratch, w.perLayerProjNorm.asFloatArray(), nLayers, nEmbdPerLayer, HEAD_NORM_LOCAL, (float) (1.0 / Math.sqrt(dim)), eps);
            g.task(name + "_plmerge", Gemma4Kernels::addAndScale, ctx, plInputs, plScratch, plTokRow, (float) (1.0 / Math.sqrt(2.0)), perLayerTotal * B);
            gs.addWorkerGrid(name + "." + name + "_embscale", ew(B * dim));
            gs.addWorkerGrid(name + "." + name + "_plmodel", ew(B * perLayerTotal));
            gs.addWorkerGrid(name + "." + name + "_plnorm", gw(B * nLayers * HEAD_NORM_LOCAL, HEAD_NORM_LOCAL));
            gs.addWorkerGrid(name + "." + name + "_plmerge", ew(B * perLayerTotal));
        } else {
            g.consumeFromDevice("L" + (l - 1), ctx, wrapX, seqPos, plTokRow, normed, normedFfn, scAtt, scPAtt, scFfn, scPFfn, scPle, qB, kB, vB, attnOut, woOut, gateUp, hb, w2Out, keyCache, valCache, plInputs, plScratch, plGate, plOut);
        }
        // per-layer weights
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                w.rms_att_weightLayered[l].asFloatArray(), w.wqLayered[l].asByteArray(), w.wkLayered[l].asByteArray(), w.wvLayered[l].asByteArray(),
                w.woLayered[l].asByteArray(), w.attnQNorm[l].asFloatArray(), w.attnKNorm[l].asFloatArray(), w.attnPostNorm[l].asFloatArray(),
                w.rms_ffn_weightLayered[l].asFloatArray(), w.w1Layered[l].asByteArray(), w.w3Layered[l].asByteArray(), w.w2Layered[l].asByteArray(),
                w.ffnPostNorm[l].asFloatArray(), w.perLayerInpGate[l].asFloatArray(), w.perLayerProj[l].asFloatArray(), w.perLayerPostNorm[l].asFloatArray(), fcr, fci);

        // ── attention ──
        g.task(name + "_anrms", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, wrapX, scAtt, dim, eps, RMS_LOCAL);
        g.task(name + "_anap", Gemma4BatchDecodeKernels::batchedGemmaApplyRmsNormFP16, ctx, normed, wrapX, w.rms_att_weightLayered[l].asFloatArray(), scAtt, dim);
        g.task(name + "_q", TransformerBatchPrefillKernels::gemmMMAQ8, ctx, normed, w.wqLayered[l].asByteArray(), qB, paddedB, qDim, dim);
        g.task(name + "_qn", Gemma4BatchDecodeKernels::batchedGemmaPerHeadRmsNorm, ctx, qB, w.attnQNorm[l].asFloatArray(), nHeads, headDim, qDim, HEAD_NORM_LOCAL, eps);
        if (own) {
            g.task(name + "_k", TransformerBatchPrefillKernels::gemmMMAQ8, ctx, normed, w.wkLayered[l].asByteArray(), kB, paddedB, kvDim, dim);
            g.task(name + "_kn", Gemma4BatchDecodeKernels::batchedGemmaPerHeadRmsNorm, ctx, kB, w.attnKNorm[l].asFloatArray(), nHeadKv, headDim, kvDim, HEAD_NORM_LOCAL, eps);
            g.task(name + "_v", TransformerBatchPrefillKernels::gemmMMAQ8, ctx, normed, w.wvLayered[l].asByteArray(), vB, paddedB, kvDim, dim);
            g.task(name + "_vn", Gemma4BatchDecodeKernels::batchedGemmaPerHeadRmsNormNoWeight, ctx, vB, nHeadKv, headDim, kvDim, HEAD_NORM_LOCAL, eps);
            g.task(name + "_rope", Gemma4BatchDecodeKernels::batchedGemmaDecodeRopeNeox, ctx, seqPos, qB, kB, vB, keyCache, valCache, fcr, fci, nHeads, nHeadKv, headDim, base, slotStride);
        } else {
            g.task(name + "_ropeq", Gemma4BatchDecodeKernels::batchedGemmaDecodeRopeQOnly, ctx, seqPos, qB, fcr, fci, nHeads, headDim, qDim);
        }
        g.task(name + "_attn", Gemma4BatchDecodeKernels::batchedGemmaDecodeAttentionFP16Out, ctx, seqPos, qB, keyCache, valCache, attnOut, nHeads, headDim, kvDim, kvMul, base, slotStride, window, qDim);
        g.task(name + "_wo", TransformerBatchPrefillKernels::gemmMMAQ8, ctx, attnOut, w.woLayered[l].asByteArray(), woOut, paddedB, dim, qDim);
        g.task(name + "_panrms", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, woOut, scPAtt, dim, eps, RMS_LOCAL);
        g.task(name + "_panap", Gemma4BatchDecodeKernels::batchedGemmaRmsNormApplyWithResidual, ctx, wrapX, woOut, w.attnPostNorm[l].asFloatArray(), scPAtt, dim);
        // ── ffn ──
        g.task(name + "_fnrms", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, wrapX, scFfn, dim, eps, RMS_LOCAL);
        g.task(name + "_fnap", Gemma4BatchDecodeKernels::batchedGemmaApplyRmsNormFP16, ctx, normedFfn, wrapX, w.rms_ffn_weightLayered[l].asFloatArray(), scFfn, dim);
        g.task(name + "_gu", TransformerBatchPrefillKernels::gemmMMAGateUpQ8, ctx, normedFfn, w.w1Layered[l].asByteArray(), w.w3Layered[l].asByteArray(), gateUp, paddedB, ffn, dim);
        g.task(name + "_geglu", Gemma4BatchDecodeKernels::batchedGemmaGeGLUPacked, ctx, hb, gateUp, ffn);
        g.task(name + "_down", TransformerBatchPrefillKernels::gemmMMAQ8, ctx, hb, w.w2Layered[l].asByteArray(), w2Out, paddedB, dim, ffn);
        g.task(name + "_pfnrms", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, w2Out, scPFfn, dim, eps, RMS_LOCAL);
        g.task(name + "_pfnap", Gemma4BatchDecodeKernels::batchedGemmaRmsNormApplyWithResidual, ctx, wrapX, w2Out, w.ffnPostNorm[l].asFloatArray(), scPFfn, dim);
        // ── PLE ──
        boolean noPle = Boolean.getBoolean("gemma.noPle");
        if (!noPle) {
        g.task(name + "_plg", Gemma4BatchDecodeKernels::batchedMatVecF32, ctx, wrapX, w.perLayerInpGate[l].asFloatArray(), plGate, dim, nEmbdPerLayer);
        g.task(name + "_plgm", Gemma4BatchDecodeKernels::batchedGemmaPleGateGeluMul, ctx, plGate, plInputs, peOff, nEmbdPerLayer, perLayerTotal);
        g.task(name + "_plp", Gemma4BatchDecodeKernels::batchedMatVecF32, ctx, plGate, w.perLayerProj[l].asFloatArray(), plOut, nEmbdPerLayer, dim);
        g.task(name + "_pprms", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, plOut, scPle, dim, eps, RMS_LOCAL);
        g.task(name + "_ppap", Gemma4BatchDecodeKernels::batchedGemmaRmsNormApplyWithResidual, ctx, wrapX, plOut, w.perLayerPostNorm[l].asFloatArray(), scPle, dim);
        }
        if (w.layerOutputScale[l] != null) {
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, w.layerOutputScale[l].asFloatArray());
            g.task(name + "_los", Gemma4Kernels::scaleInPlaceFromTensor, ctx, wrapX, w.layerOutputScale[l].asFloatArray(), B * dim);
            gs.addWorkerGrid(name + "." + name + "_los", ew(B * dim));
        }
        if (Boolean.getBoolean("gemma.dbgL0")) g.transferToHost(DataTransferMode.EVERY_EXECUTION, wrapX);
        g.persistOnDevice(wrapX, keyCache, valCache, plInputs);

        // workers
        WorkerGrid rms = gw(B * RMS_LOCAL, RMS_LOCAL), ap = ew(B * dim);
        gs.addWorkerGrid(name + "." + name + "_anrms", rms); gs.addWorkerGrid(name + "." + name + "_anap", ap);
        gs.addWorkerGrid(name + "." + name + "_q", mma(paddedB, qDim));
        gs.addWorkerGrid(name + "." + name + "_qn", gw(B * nHeads * HEAD_NORM_LOCAL, HEAD_NORM_LOCAL));
        if (own) {
            gs.addWorkerGrid(name + "." + name + "_k", mma(paddedB, kvDim)); gs.addWorkerGrid(name + "." + name + "_kn", gw(B * nHeadKv * HEAD_NORM_LOCAL, HEAD_NORM_LOCAL));
            gs.addWorkerGrid(name + "." + name + "_v", mma(paddedB, kvDim)); gs.addWorkerGrid(name + "." + name + "_vn", gw(B * nHeadKv * HEAD_NORM_LOCAL, HEAD_NORM_LOCAL));
            gs.addWorkerGrid(name + "." + name + "_rope", ew(B * nHeads * (headDim / 2)));
        } else {
            gs.addWorkerGrid(name + "." + name + "_ropeq", ew(B * nHeads * (headDim / 2)));
        }
        int attnLocal = Math.min(headDim, 128);
        gs.addWorkerGrid(name + "." + name + "_attn", gw(B * nHeads * attnLocal, attnLocal));
        gs.addWorkerGrid(name + "." + name + "_wo", mma(paddedB, dim));
        gs.addWorkerGrid(name + "." + name + "_panrms", rms); gs.addWorkerGrid(name + "." + name + "_panap", ap);
        gs.addWorkerGrid(name + "." + name + "_fnrms", rms); gs.addWorkerGrid(name + "." + name + "_fnap", ap);
        gs.addWorkerGrid(name + "." + name + "_gu", mma(paddedB, ffn)); gs.addWorkerGrid(name + "." + name + "_geglu", ew(B * ffn));
        gs.addWorkerGrid(name + "." + name + "_down", mma(paddedB, dim));
        gs.addWorkerGrid(name + "." + name + "_pfnrms", rms); gs.addWorkerGrid(name + "." + name + "_pfnap", ap);
        gs.addWorkerGrid(name + "." + name + "_plg", ew(B * nEmbdPerLayer)); gs.addWorkerGrid(name + "." + name + "_plgm", ew(B * nEmbdPerLayer));
        gs.addWorkerGrid(name + "." + name + "_plp", ew(B * dim));
        gs.addWorkerGrid(name + "." + name + "_pprms", rms); gs.addWorkerGrid(name + "." + name + "_ppap", ap);
        return g;
    }

    TaskGraph buildLogits(GridScheduler gs) {
        String name = "LOGITS";
        TaskGraph g = new TaskGraph(name);
        g.consumeFromDevice("L" + (nLayers - 1), ctx, wrapX);
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, ctx, normedFinal, scLog, sampled, w.wclsByteArray.asByteArray(), w.rms_final_weight_as_floatArray.asFloatArray());
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, wrapX);
        g.task(name + "_rms", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, wrapX, scLog, dim, eps, RMS_LOCAL);
        g.task(name + "_ap", Gemma4BatchDecodeKernels::batchedGemmaApplyRmsNormFP16, ctx, normedFinal, wrapX, w.rms_final_weight_as_floatArray.asFloatArray(), scLog, dim);
        g.task(name + "_vocab", TransformerBatchPrefillKernels::gemmMMAQ8, ctx, normedFinal, w.wclsByteArray.asByteArray(), logits, paddedB, vocab, dim);
        g.task(name + "_argmax", TransformerBatchPrefillKernels::batchedArgmaxLogits, ctx, logits, sampled, vocab);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, sampled, logits);
        gs.addWorkerGrid(name + "." + name + "_rms", gw(B * RMS_LOCAL, RMS_LOCAL));
        gs.addWorkerGrid(name + "." + name + "_ap", ew(B * dim));
        gs.addWorkerGrid(name + "." + name + "_vocab", mma(paddedB, vocab));
        gs.addWorkerGrid(name + "." + name + "_argmax", gw(B * 256, 256));
        return g;
    }

    void execGraph(TornadoExecutionPlan plan, GridScheduler gs, int idx, boolean cudaGraphs) {
        var e = plan.withGraph(idx).withGridScheduler(gs);
        if (cudaGraphs) e.withCUDAGraph();
        e.execute();
    }

    static WorkerGrid ew(int n) { WorkerGrid1D g = new WorkerGrid1D(n); g.setLocalWork(Math.min(256, n), 1, 1); return g; }
    static WorkerGrid gw(int global, int local) { WorkerGrid1D g = new WorkerGrid1D(global); g.setLocalWork(local, 1, 1); return g; }
    static WorkerGrid mma(int m, int n) { WorkerGrid2D g = new WorkerGrid2D((m / 128) * 256, n / 128); g.setLocalWork(256, 1, 1); return g; }
}
