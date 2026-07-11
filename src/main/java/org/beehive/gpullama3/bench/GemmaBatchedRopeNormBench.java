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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.Random;

/** Validates the batched Gemma per-head RMSNorm and NEOX RoPE (own-KV) kernels vs CPU. */
public class GemmaBatchedRopeNormBench {

    static final int N_HEADS = 8, HEAD_DIM = 256, N_KV_HEADS = 2;
    static final int KV_DIM = N_KV_HEADS * HEAD_DIM, Q_DIM = N_HEADS * HEAD_DIM;
    static final int N_LAYERS = 1, CTX = 1024, LAYER = 0, HALF = HEAD_DIM / 2;
    static final int NORM_LOCAL = 64;
    static final float EPS = 1e-6f;

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int B = args.length > 0 ? Integer.parseInt(args[0]) : 32;
        int pos = args.length > 1 ? Integer.parseInt(args[1]) : 137;
        Random rnd = new Random(5);

        // ── per-head norm (weighted) on Q ──
        FloatArray q = new FloatArray(B * Q_DIM);
        FloatArray qn = new FloatArray(B * Q_DIM);
        FloatArray wNorm = new FloatArray(HEAD_DIM);
        for (int i = 0; i < B * Q_DIM; i++) { float v = rnd.nextFloat() - 0.5f; q.set(i, v); qn.set(i, v); }
        for (int i = 0; i < HEAD_DIM; i++) wNorm.set(i, rnd.nextFloat() + 0.5f);

        KernelContext ctx = new KernelContext();
        WorkerGrid1D nw = new WorkerGrid1D(B * N_HEADS * NORM_LOCAL);
        nw.setLocalWork(NORM_LOCAL, 1, 1);
        GridScheduler ng = new GridScheduler("n.norm", nw);
        TaskGraph nt = new TaskGraph("n")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, qn, wNorm)
                .task("norm", Gemma4BatchDecodeKernels::batchedGemmaPerHeadRmsNorm,
                        ctx, qn, wNorm, N_HEADS, HEAD_DIM, Q_DIM, NORM_LOCAL, EPS)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, qn);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(nt.snapshot())) {
            plan.withGridScheduler(ng);
            plan.execute();
        }
        double normMax = 0;
        for (int b = 0; b < B; b++) for (int h = 0; h < N_HEADS; h++) {
            int base = b * Q_DIM + h * HEAD_DIM;
            double ss = 0; for (int i = 0; i < HEAD_DIM; i++) { float v = q.get(base + i); ss += v * v; }
            float inv = (float) (1.0 / Math.sqrt(ss / HEAD_DIM + EPS));
            for (int i = 0; i < HEAD_DIM; i++) {
                float ref = wNorm.get(i) * (inv * q.get(base + i));
                normMax = Math.max(normMax, Math.abs(ref - qn.get(base + i)) / Math.max(1e-4, Math.abs(ref)));
            }
        }

        // ── NEOX rope (own-KV) ──
        FloatArray q2 = new FloatArray(B * Q_DIM), k = new FloatArray(B * KV_DIM), v = new FloatArray(B * KV_DIM);
        FloatArray keyCache = new FloatArray(B * N_LAYERS * CTX * KV_DIM), valCache = new FloatArray(B * N_LAYERS * CTX * KV_DIM);
        FloatArray fcr = new FloatArray(CTX * HALF), fci = new FloatArray(CTX * HALF);
        IntArray seqPos = new IntArray(B);
        float[] q2ref = new float[B * Q_DIM];
        for (int i = 0; i < B * Q_DIM; i++) { float x = rnd.nextFloat() - 0.5f; q2.set(i, x); q2ref[i] = x; }
        for (int i = 0; i < B * KV_DIM; i++) { k.set(i, rnd.nextFloat() - 0.5f); v.set(i, rnd.nextFloat() - 0.5f); }
        for (int i = 0; i < CTX * HALF; i++) { fcr.set(i, (float) Math.cos(i * 0.001)); fci.set(i, (float) Math.sin(i * 0.001)); }
        for (int b = 0; b < B; b++) seqPos.set(b, pos);

        KernelContext ctx2 = new KernelContext();
        WorkerGrid1D rw = new WorkerGrid1D(B * N_HEADS * HALF);
        rw.setLocalWork(HALF, 1, 1);
        GridScheduler rg = new GridScheduler("r.rope", rw);
        TaskGraph rt = new TaskGraph("r")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, q2, k, v, seqPos, fcr, fci)
                .task("rope", Gemma4BatchDecodeKernels::batchedGemmaDecodeRopeNeox,
                        ctx2, seqPos, q2, k, v, keyCache, valCache, fcr, fci,
                        N_HEADS, N_KV_HEADS, HEAD_DIM, LAYER * CTX * KV_DIM, N_LAYERS * CTX * KV_DIM)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, q2, keyCache, valCache);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(rt.snapshot())) {
            plan.withGridScheduler(rg);
            plan.execute();
        }
        double ropeQMax = 0, ropeKMax = 0, ropeVMax = 0;
        float[] kref = new float[B * KV_DIM];
        for (int i = 0; i < B * KV_DIM; i++) kref[i] = k.get(i);   // k already rotated on device; recompute from scratch below
        // recompute CPU refs from the ORIGINAL k? k was mutated on host? No — k is a device buffer, host copy unchanged except transferToHost didn't include k. Use original via re-derivation:
        Random rnd2 = new Random(5);
        // skip Q-norm consumption of rnd to realign — instead just reconstruct k,v deterministically:
        // (k,v were filled after q2/q2ref which consumed B*Q_DIM; replicate that consumption)
        for (int i = 0; i < B * Q_DIM; i++) rnd2.nextFloat();          // q(norm) block
        for (int i = 0; i < HEAD_DIM; i++) rnd2.nextFloat();           // wNorm
        for (int i = 0; i < B * Q_DIM; i++) rnd2.nextFloat();          // q2
        float[] k0 = new float[B * KV_DIM], v0 = new float[B * KV_DIM];
        for (int i = 0; i < B * KV_DIM; i++) { k0[i] = rnd2.nextFloat() - 0.5f; v0[i] = rnd2.nextFloat() - 0.5f; }
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < N_HEADS; h++) {
                int qBase = b * Q_DIM + h * HEAD_DIM;
                for (int ic = 0; ic < HALF; ic++) {
                    float c = (float) Math.cos((pos * HALF + ic) * 0.001), s = (float) Math.sin((pos * HALF + ic) * 0.001);
                    float a = q2ref[qBase + ic], bb = q2ref[qBase + ic + HALF];
                    float e0 = a * c - bb * s, e1 = a * s + bb * c;
                    ropeQMax = Math.max(ropeQMax, Math.abs(e0 - q2.get(qBase + ic)) / Math.max(1e-4, Math.abs(e0)));
                    ropeQMax = Math.max(ropeQMax, Math.abs(e1 - q2.get(qBase + ic + HALF)) / Math.max(1e-4, Math.abs(e1)));
                }
            }
            long cbase = (long) b * N_LAYERS * CTX * KV_DIM + (long) pos * KV_DIM;
            for (int h = 0; h < N_KV_HEADS; h++) {
                int kBase = b * KV_DIM + h * HEAD_DIM;
                for (int ic = 0; ic < HALF; ic++) {
                    float c = (float) Math.cos((pos * HALF + ic) * 0.001), s = (float) Math.sin((pos * HALF + ic) * 0.001);
                    float a = k0[kBase + ic], bb = k0[kBase + ic + HALF];
                    float e0 = a * c - bb * s, e1 = a * s + bb * c;
                    ropeKMax = Math.max(ropeKMax, Math.abs(e0 - keyCache.get((int) (cbase + h * HEAD_DIM + ic))) / Math.max(1e-4, Math.abs(e0)));
                    ropeKMax = Math.max(ropeKMax, Math.abs(e1 - keyCache.get((int) (cbase + h * HEAD_DIM + ic + HALF))) / Math.max(1e-4, Math.abs(e1)));
                    ropeVMax = Math.max(ropeVMax, Math.abs(v0[kBase + ic] - valCache.get((int) (cbase + h * HEAD_DIM + ic))));
                }
            }
        }
        System.out.printf("Gemma per-head RMSNorm maxRel=%.5f%n", normMax);
        System.out.printf("Gemma NEOX rope: Q maxRel=%.5f  Kcache maxRel=%.5f  Vcache maxAbs=%.5f%n", ropeQMax, ropeKMax, ropeVMax);
    }
}
