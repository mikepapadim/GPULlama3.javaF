package org.beehive.gpullama3.bench;

import org.beehive.gpullama3.tornadovm.kernels.Gemma4BatchDecodeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.util.Random;

/** Validates batched Gemma GeGLU + RMSNorm-apply + norm-apply-with-residual vs CPU. */
public class GemmaBatchedFFNBench {

    static final int DIM = 2048, HIDDEN = 4096, RMS_LOCAL = 256;
    static final float EPS = 1e-6f;

    static float gelu(float g) {
        float g3 = g * g * g;
        return 0.5f * g * (1.0f + (float) Math.tanh(0.797885f * (g + 0.044715f * g3)));
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int B = args.length > 0 ? Integer.parseInt(args[0]) : 32;
        Random rnd = new Random(9);
        KernelContext ctx = new KernelContext();

        // ── GeGLU ──
        FloatArray gateUp = new FloatArray(B * 2 * HIDDEN);
        HalfFloatArray hb = new HalfFloatArray(B * HIDDEN);
        for (int i = 0; i < B * 2 * HIDDEN; i++) gateUp.set(i, rnd.nextFloat() - 0.5f);
        WorkerGrid1D gw = new WorkerGrid1D(B * HIDDEN);
        gw.setLocalWork(256, 1, 1);
        GridScheduler gg = new GridScheduler("f.geglu", gw);
        TaskGraph ft = new TaskGraph("f")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, gateUp)
                .task("geglu", Gemma4BatchDecodeKernels::batchedGemmaGeGLUPacked, ctx, hb, gateUp, HIDDEN)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, hb);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(ft.snapshot())) {
            plan.withGridScheduler(gg); plan.execute();
        }
        double geMax = 0;
        for (int b = 0; b < B; b++) for (int i = 0; i < HIDDEN; i++) {
            int rb = b * 2 * HIDDEN;
            float ref = gelu(gateUp.get(rb + i)) * gateUp.get(rb + HIDDEN + i);
            geMax = Math.max(geMax, Math.abs(ref - hb.get(b * HIDDEN + i).getFloat32()) / Math.max(1e-3, Math.abs(ref)));
        }

        // ── RMSNorm apply + norm-with-residual ──
        FloatArray x = new FloatArray(B * DIM), out = new FloatArray(B * DIM);
        FloatArray delta = new FloatArray(B * DIM), xres = new FloatArray(B * DIM);
        FloatArray weight = new FloatArray(DIM), scale = new FloatArray(B);
        float[] x0 = new float[B * DIM], xres0 = new float[B * DIM], delta0 = new float[B * DIM];
        for (int i = 0; i < B * DIM; i++) { float v = rnd.nextFloat() - 0.5f; x.set(i, v); x0[i] = v; float r = rnd.nextFloat() - 0.5f; xres.set(i, r); xres0[i] = r; float d = rnd.nextFloat() - 0.5f; delta.set(i, d); delta0[i] = d; }
        for (int i = 0; i < DIM; i++) weight.set(i, rnd.nextFloat() + 0.5f);

        WorkerGrid1D rw = new WorkerGrid1D(B * RMS_LOCAL); rw.setLocalWork(RMS_LOCAL, 1, 1);
        WorkerGrid1D aw = new WorkerGrid1D(B * DIM); aw.setLocalWork(256, 1, 1);
        GridScheduler ng = new GridScheduler();
        ng.addWorkerGrid("n.reduce", rw); ng.addWorkerGrid("n.apply", aw); ng.addWorkerGrid("n.resid", aw); ng.addWorkerGrid("n.reduce2", rw);
        TaskGraph nt = new TaskGraph("n")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, delta, xres, weight)
                .task("reduce", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, x, scale, DIM, EPS, RMS_LOCAL)
                .task("apply", Gemma4BatchDecodeKernels::batchedGemmaApplyRmsNorm, ctx, out, x, weight, scale, DIM)
                .task("reduce2", TransformerBatchPrefillKernels::batchedRmsReduceParallel, ctx, delta, scale, DIM, EPS, RMS_LOCAL)
                .task("resid", Gemma4BatchDecodeKernels::batchedGemmaRmsNormApplyWithResidual, ctx, xres, delta, weight, scale, DIM)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out, xres);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(nt.snapshot())) {
            plan.withGridScheduler(ng); plan.execute();
        }
        double applyMax = 0, residMax = 0;
        for (int b = 0; b < B; b++) {
            double ss = 0; for (int i = 0; i < DIM; i++) ss += x0[b * DIM + i] * x0[b * DIM + i];
            float sc = (float) (1.0 / Math.sqrt(ss / DIM + EPS));
            for (int i = 0; i < DIM; i++) {
                float ref = weight.get(i) * (sc * x0[b * DIM + i]);
                applyMax = Math.max(applyMax, Math.abs(ref - out.get(b * DIM + i)) / Math.max(1e-3, Math.abs(ref)));
            }
            double ssd = 0; for (int i = 0; i < DIM; i++) ssd += delta0[b * DIM + i] * delta0[b * DIM + i];
            float scd = (float) (1.0 / Math.sqrt(ssd / DIM + EPS));
            for (int i = 0; i < DIM; i++) {
                float ref = xres0[b * DIM + i] + weight.get(i) * (scd * delta0[b * DIM + i]);
                residMax = Math.max(residMax, Math.abs(ref - xres.get(b * DIM + i)) / Math.max(1e-3, Math.abs(ref)));
            }
        }
        System.out.printf("Gemma GeGLU maxRel=%.4f | RMSNorm-apply maxRel=%.5f | norm+residual maxRel=%.5f%n", geMax, applyMax, residMax);
    }
}
