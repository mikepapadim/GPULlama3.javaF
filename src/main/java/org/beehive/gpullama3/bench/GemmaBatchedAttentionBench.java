package org.beehive.gpullama3.bench;

import org.beehive.gpullama3.tornadovm.kernels.Gemma4BatchDecodeKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.Random;

/**
 * Standalone validation of {@link Gemma4BatchDecodeKernels#batchedGemmaDecodeAttentionFP16Out}
 * against a CPU reference: B independent sequences, per-slot KV, sliding-window causal attention
 * with Gemma's scale 1.0. Synthetic dims (no model load).
 *
 * Run: ... GemmaBatchedAttentionBench [B] [seqLen] [windowSize]
 */
public class GemmaBatchedAttentionBench {

    static final int N_HEADS = 8;
    static int HEAD_DIM = 256;
    static final int N_KV_HEADS = 2;
    static int KV_DIM = N_KV_HEADS * HEAD_DIM;
    static final int KV_MUL = N_HEADS / N_KV_HEADS;      // 4
    static int Q_DIM = N_HEADS * HEAD_DIM;
    static final int N_LAYERS = 1;
    static final int CTX = 1024;
    static final int LAYER = 0;
    static final int LOCAL = 128;                        // min(headDim,128)

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int B = args.length > 0 ? Integer.parseInt(args[0]) : 32;
        int seqLen = args.length > 1 ? Integer.parseInt(args[1]) : 300;
        int windowSize = args.length > 2 ? Integer.parseInt(args[2]) : 128;   // < seqLen → exercise windowing
        HEAD_DIM = args.length > 3 ? Integer.parseInt(args[3]) : 256;         // 256 (swa) / 512 (full)
        KV_DIM = N_KV_HEADS * HEAD_DIM;
        Q_DIM = N_HEADS * HEAD_DIM;
        Random rnd = new Random(11);

        FloatArray q = new FloatArray(B * Q_DIM);
        FloatArray keyCache = new FloatArray(B * N_LAYERS * CTX * KV_DIM);
        FloatArray valueCache = new FloatArray(B * N_LAYERS * CTX * KV_DIM);
        HalfFloatArray xb = new HalfFloatArray(B * Q_DIM);
        IntArray seqPos = new IntArray(B);

        for (int i = 0; i < B * Q_DIM; i++) {
            q.set(i, rnd.nextFloat() - 0.5f);
        }
        for (int b = 0; b < B; b++) {
            seqPos.set(b, seqLen - 1);
            long base = (long) b * N_LAYERS * CTX * KV_DIM;
            for (int t = 0; t < seqLen; t++) {
                for (int d = 0; d < KV_DIM; d++) {
                    keyCache.set((int) (base + (long) t * KV_DIM + d), rnd.nextFloat() - 0.5f);
                    valueCache.set((int) (base + (long) t * KV_DIM + d), rnd.nextFloat() - 0.5f);
                }
            }
        }

        KernelContext ctx = new KernelContext();
        WorkerGrid1D worker = new WorkerGrid1D(B * N_HEADS * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        GridScheduler grid = new GridScheduler("g.attn", worker);
        TaskGraph tg = new TaskGraph("g")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, q, keyCache, valueCache, seqPos)
                .task("attn", Gemma4BatchDecodeKernels::batchedGemmaDecodeAttentionFP16Out,
                        ctx, seqPos, q, keyCache, valueCache, xb,
                        N_HEADS, HEAD_DIM, KV_DIM, KV_MUL, LAYER * CTX * KV_DIM, N_LAYERS * CTX * KV_DIM, windowSize, Q_DIM)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(grid);
            for (int i = 0; i < 5; i++) {
                plan.execute();
            }
            float[] ref = cpuReference(q, keyCache, valueCache, seqPos, B, windowSize);
            double maxRel = 0.0;
            int bad = 0;
            for (int i = 0; i < B * Q_DIM; i++) {
                float e = ref[i];
                float a = xb.get(i).getFloat32();
                double rel = Math.abs(e - a) / Math.max(1e-3, Math.abs(e));
                maxRel = Math.max(maxRel, rel);
                if (rel > 3e-2) {                            // FP16 output tolerance
                    bad++;
                }
            }
            System.out.printf("Gemma batched windowed attention: B=%d seqLen=%d window=%d  maxRel=%.4f out-of-tol=%d/%d%n",
                    B, seqLen, windowSize, maxRel, bad, B * Q_DIM);
        }
    }

    private static float[] cpuReference(FloatArray q, FloatArray keyCache, FloatArray valueCache, IntArray seqPos, int B, int windowSize) {
        float[] out = new float[B * Q_DIM];
        for (int b = 0; b < B; b++) {
            int pos = seqPos.get(b);
            int windowStart = Math.max(0, pos - windowSize + 1);
            long base = (long) b * N_LAYERS * CTX * KV_DIM;
            for (int h = 0; h < N_HEADS; h++) {
                int kvHead = h / KV_MUL;
                int qOff = b * Q_DIM + h * HEAD_DIM;
                float[] scores = new float[pos + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int t = windowStart; t <= pos; t++) {
                    float s = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        s += q.get(qOff + d) * keyCache.get((int) (base + (long) t * KV_DIM + kvHead * HEAD_DIM + d));
                    }
                    scores[t] = s;                            // scale 1.0
                    max = Math.max(max, s);
                }
                float sum = 0.0f;
                for (int t = windowStart; t <= pos; t++) {
                    scores[t] = (float) Math.exp(scores[t] - max);
                    sum += scores[t];
                }
                float inv = sum > 0 ? 1.0f / sum : 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    float acc = 0.0f;
                    for (int t = windowStart; t <= pos; t++) {
                        acc += scores[t] * inv * valueCache.get((int) (base + (long) t * KV_DIM + kvHead * HEAD_DIM + d));
                    }
                    out[qOff + d] = acc;
                }
            }
        }
        return out;
    }
}
