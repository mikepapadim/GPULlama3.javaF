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

import java.util.Random;

/** Validates batched Gemma PLE kernels (gate-gelu-mul + per-segment scale-normalize) vs CPU. */
public class GemmaBatchedPleBench {

    static final int PE = 256, N_LAYERS = 8, SEG = 256, LOCAL = 64;
    static final int TOTAL = N_LAYERS * PE;
    static final float PRE = 0.75f, EPS = 1e-6f;

    static float gelu(float g) {
        float g3 = g * g * g;
        return 0.5f * g * (1.0f + (float) Math.tanh(0.797885f * (g + 0.044715f * g3)));
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int B = args.length > 0 ? Integer.parseInt(args[0]) : 32;
        int peOffset = 3 * PE;                    // pretend layer 3
        Random rnd = new Random(3);
        KernelContext ctx = new KernelContext();

        // ── gate gelu mul ──
        FloatArray gate = new FloatArray(B * PE), inputs = new FloatArray(B * TOTAL);
        float[] gate0 = new float[B * PE];
        for (int i = 0; i < B * PE; i++) { float v = rnd.nextFloat() - 0.5f; gate.set(i, v); gate0[i] = v; }
        for (int i = 0; i < B * TOTAL; i++) inputs.set(i, rnd.nextFloat() - 0.5f);
        WorkerGrid1D gw = new WorkerGrid1D(B * PE); gw.setLocalWork(256, 1, 1);
        GridScheduler gg = new GridScheduler("p.gate", gw);
        TaskGraph gt = new TaskGraph("p")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, gate, inputs)
                .task("gate", Gemma4BatchDecodeKernels::batchedGemmaPleGateGeluMul, ctx, gate, inputs, peOffset, PE, TOTAL)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, gate);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(gt.snapshot())) { plan.withGridScheduler(gg); plan.execute(); }
        double gateMax = 0;
        for (int b = 0; b < B; b++) for (int i = 0; i < PE; i++) {
            float ref = gelu(gate0[b * PE + i]) * inputs.get(b * TOTAL + peOffset + i);
            gateMax = Math.max(gateMax, Math.abs(ref - gate.get(b * PE + i)) / Math.max(1e-3, Math.abs(ref)));
        }

        // ── per-segment scale + normalize ──
        FloatArray x = new FloatArray(B * TOTAL), weight = new FloatArray(SEG);
        float[] x0 = new float[B * TOTAL];
        for (int i = 0; i < B * TOTAL; i++) { float v = rnd.nextFloat() - 0.5f; x.set(i, v); x0[i] = v; }
        for (int i = 0; i < SEG; i++) weight.set(i, rnd.nextFloat() + 0.5f);
        WorkerGrid1D sw = new WorkerGrid1D(B * N_LAYERS * LOCAL); sw.setLocalWork(LOCAL, 1, 1);
        GridScheduler sg = new GridScheduler("s.norm", sw);
        TaskGraph st = new TaskGraph("s")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, weight)
                .task("norm", Gemma4BatchDecodeKernels::batchedGemmaPleProjScaleAndNormalize, ctx, x, weight, N_LAYERS, SEG, LOCAL, PRE, EPS)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, x);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(st.snapshot())) { plan.withGridScheduler(sg); plan.execute(); }
        double segMax = 0;
        for (int b = 0; b < B; b++) for (int seg = 0; seg < N_LAYERS; seg++) {
            int base = b * TOTAL + seg * SEG;
            double ss = 0; for (int i = 0; i < SEG; i++) { float v = x0[base + i] * PRE; ss += v * v; }
            float sc = (float) (1.0 / Math.sqrt(ss / SEG + EPS));
            for (int i = 0; i < SEG; i++) {
                float ref = weight.get(i) * (sc * (x0[base + i] * PRE));
                segMax = Math.max(segMax, Math.abs(ref - x.get(base + i)) / Math.max(1e-3, Math.abs(ref)));
            }
        }
        System.out.printf("Gemma PLE: gate-gelu-mul maxRel=%.5f | proj-scale-normalize maxRel=%.5f%n", gateMax, segMax);
    }
}
