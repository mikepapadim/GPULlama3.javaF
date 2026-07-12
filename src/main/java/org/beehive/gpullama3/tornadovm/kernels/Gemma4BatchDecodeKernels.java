package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Batched-DECODE kernels for the Gemma 4 architecture (B independent sequences, one token/step,
 * each with its own KV region). Gemma 4 differs from Llama/Qwen3 enough that its batched-decode
 * layer graph needs its own kernels; this class collects the Gemma-specific ones as they are
 * ported and validated against the single-token {@link Gemma4Kernels} reference.
 *
 * <p>Port status (see GEMMA4_BATCHED_PLAN.md):</p>
 * <ul>
 *   <li>[x] {@link #batchedGemmaDecodeAttentionFP16Out} — sliding-window / full attention, scale 1.0</li>
 *   <li>[ ] batched NEOX RoPE + per-slot KV write (own-KV vs Q-only layers)</li>
 *   <li>[ ] batched per-head Q/K/V RMSNorm</li>
 *   <li>[ ] batched GeGLU (Q8 gate/up)</li>
 *   <li>[ ] batched pre/post RMSNorm (+residual), PLE tasks, logit softcap</li>
 * </ul>
 */
public final class Gemma4BatchDecodeKernels {

    private Gemma4BatchDecodeKernels() {
    }

    /**
     * Batched per-slot windowed flash attention, FP16 output (for the Q8 Wo MMA GEMM).
     *
     * <p>Mirrors {@link Gemma4Kernels#attentionWithSlidingWindow}: each (slot, head) attends over
     * {@code t ∈ [max(0, pos-windowSize+1), pos]} of its own KV region, with Gemma's attention
     * scale of {@code 1.0} (no {@code 1/sqrt(headDim)}). Full-attention layers pass
     * {@code windowSize >= contextLength}. Query is FP32 ({@code qBatch}, already norm+RoPE'd),
     * KV cache is FP32 and per-slot (stride {@code numLayers*contextLength*kvDim}); output is FP16.</p>
     *
     * <p>Requires headDim &le; 2*localSz (localSz = min(headDim, 128)); headDim &le; 256.</p>
     *
     * <p>Worker: B*nHeads workgroups × min(headDim,128) threads.</p>
     */
    public static void batchedGemmaDecodeAttentionFP16Out(KernelContext context,
                                                          IntArray seqPositions,
                                                          FloatArray qBatch,
                                                          FloatArray keyCache,
                                                          FloatArray valueCache,
                                                          HalfFloatArray attnOutFP16,
                                                          int nHeads, int headDim,
                                                          int kvDim, int kvMul,
                                                          int layerBaseOff, int slotStride,
                                                          int windowSize, int qDim) {
        // Gemma KV cache is per-slot (slotStride = single-seq total own-KV elements) with a
        // per-layer base (layerBaseOff = cacheLayerBaseOffset[attnLayer]); reuse-KV layers pass
        // the base of the layer whose cache they share. kvDim/headDim are per-layer.
        int tid = context.localIdx;
        int groupId = context.groupIdx;
        int localSz = context.localGroupSizeX;

        int batchIdx = groupId / nHeads;
        int h = groupId % nHeads;
        int pos = seqPositions.get(batchIdx);
        int windowStart = Math.max(0, pos - windowSize + 1);
        int loff = batchIdx * slotStride + layerBaseOff;
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 8;                                          // 8*512 tiles keep shared mem ~35 KB

        // headDim up to 512 (Gemma full-attention layers), localSz = 128 → 4 output dims/thread.
        float[] qShared = context.allocateFloatLocalArray(512);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * 512);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * 512);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);

        int qOffset = batchIdx * qDim + h * headDim;
        for (int i = tid; i < headDim; i += localSz) {
            qShared[i] = qBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        int d1 = tid + localSz, d2 = tid + 2 * localSz, d3 = tid + 3 * localSz;

        for (int tileC = windowStart; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);
            int tileLen = tileEnd - tileC + 1;

            for (int idx = tid; idx < tileLen * headDim; idx += localSz) {
                int tInTile = idx / headDim;
                int d = idx % headDim;
                int kvOff = loff + (tileC + tInTile) * kvDim + kvHeadIdx * headDim + d;
                kTile[tInTile * headDim + d] = keyCache.get(kvOff);
                vTile[tInTile * headDim + d] = valueCache.get(kvOff);
            }
            context.localBarrier();

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += qShared[d] * kTile[tInTile * headDim + d];
                }
                sTile[tInTile] = score;                          // Gemma scale = 1.0
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < tileLen; t++) {
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }

            float newMax = Math.max(maxScore, tileMax);
            if (maxScore != Float.NEGATIVE_INFINITY && newMax != maxScore) {
                float corr = TornadoMath.exp(maxScore - newMax);
                sumExp *= corr;
                acc0 *= corr; acc1 *= corr; acc2 *= corr; acc3 *= corr;
            }
            maxScore = newMax;

            for (int t = 0; t < tileLen; t++) {
                float p = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += p;
                int vb = t * headDim;
                acc0 += p * vTile[vb + tid];
                if (d1 < headDim) acc1 += p * vTile[vb + d1];
                if (d2 < headDim) acc2 += p * vTile[vb + d2];
                if (d3 < headDim) acc3 += p * vTile[vb + d3];
            }
            context.localBarrier();
        }

        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int outOffset = batchIdx * qDim + h * headDim;
        attnOutFP16.set(outOffset + tid, new HalfFloat(acc0 * norm));
        if (d1 < headDim) attnOutFP16.set(outOffset + d1, new HalfFloat(acc1 * norm));
        if (d2 < headDim) attnOutFP16.set(outOffset + d2, new HalfFloat(acc2 * norm));
        if (d3 < headDim) attnOutFP16.set(outOffset + d3, new HalfFloat(acc3 * norm));
    }

    // ── Per-head RMSNorm (batched) ───────────────────────────────────────────

    /**
     * Batched per-head RMSNorm with a learned scale (Gemma Q/K norm). One workgroup per
     * (slot, head); {@code rowStride} is qDim (Q) or kvDim (K). Fork of
     * {@link Gemma4Kernels#rmsNormPerHead}. Worker: B*nHeads workgroups × localMemSize threads.
     */
    public static void batchedGemmaPerHeadRmsNorm(KernelContext context, FloatArray vecBatch, FloatArray weight,
                                                  int nHeads, int headDim, int rowStride, int localMemSize, float eps) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int batchIdx = groupId / nHeads;
        int headIdx = groupId % nHeads;
        int base = batchIdx * rowStride + headIdx * headDim;

        float[] localSum = context.allocateFloatLocalArray(64);
        float partial = 0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = vecBatch.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = 1.0f / TornadoMath.sqrt(localSum[0] / headDim + eps);
        context.localBarrier();
        for (int i = localId; i < headDim; i += localSize) {
            vecBatch.set(base + i, weight.get(i) * (ss * vecBatch.get(base + i)));
        }
    }

    /** Batched per-head RMSNorm without a learned scale (Gemma V norm). */
    public static void batchedGemmaPerHeadRmsNormNoWeight(KernelContext context, FloatArray vecBatch,
                                                          int nHeads, int headDim, int rowStride, int localMemSize, float eps) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int batchIdx = groupId / nHeads;
        int headIdx = groupId % nHeads;
        int base = batchIdx * rowStride + headIdx * headDim;

        float[] localSum = context.allocateFloatLocalArray(64);
        float partial = 0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = vecBatch.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = 1.0f / TornadoMath.sqrt(localSum[0] / headDim + eps);
        context.localBarrier();
        for (int i = localId; i < headDim; i += localSize) {
            vecBatch.set(base + i, ss * vecBatch.get(base + i));
        }
    }

    // ── NEOX RoPE (batched decode) ───────────────────────────────────────────

    /**
     * Batched NEOX RoPE + per-slot KV write for own-KV layers. Fork of
     * {@link Gemma4Kernels#ropeNeoxRotateAndCacheCopy}: rotates Q (all heads) and K (KV heads),
     * writes rotated-K / raw-V into the slot's own KV region at its own position.
     * Worker: B*nHeads*(headDim/2) global threads.
     */
    public static void batchedGemmaDecodeRopeNeox(KernelContext context,
                                                  IntArray seqPositions,
                                                  FloatArray qBatch, FloatArray kBatch, FloatArray vBatch,
                                                  FloatArray keyCache, FloatArray valueCache,
                                                  FloatArray freqCisReal, FloatArray freqCisImag,
                                                  int nHeads, int nHeadKv, int headDim,
                                                  int layerOff, int slotStride) {
        // qDim = nHeads*headDim, kvDim = nHeadKv*headDim derived to stay within the task arg limit;
        // layerOff = layerIndex*contextLength*kvDim, slotStride = numLayers*contextLength*kvDim.
        int qDim = nHeads * headDim;
        int kvDim = nHeadKv * headDim;
        int half = headDim / 2;
        int g = context.globalIdx;
        int batchIdx = g / (nHeads * half);
        int rem = g % (nHeads * half);
        int h = rem / half;
        int ic = rem % half;

        int pos = seqPositions.get(batchIdx);
        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int qBase = batchIdx * qDim + h * headDim;
        float v0q = qBatch.get(qBase + ic);
        float v1q = qBatch.get(qBase + ic + half);
        qBatch.set(qBase + ic, v0q * fcr - v1q * fci);
        qBatch.set(qBase + ic + half, v0q * fci + v1q * fcr);

        if (h < nHeadKv) {
            int kBase = batchIdx * kvDim + h * headDim;
            float v0k = kBatch.get(kBase + ic);
            float v1k = kBatch.get(kBase + ic + half);
            float rotK0 = v0k * fcr - v1k * fci;
            float rotK1 = v0k * fci + v1k * fcr;
            kBatch.set(kBase + ic, rotK0);
            kBatch.set(kBase + ic + half, rotK1);

            int cacheOff = batchIdx * slotStride + layerOff + pos * kvDim + h * headDim;
            keyCache.set(cacheOff + ic, rotK0);
            keyCache.set(cacheOff + ic + half, rotK1);
            valueCache.set(cacheOff + ic, vBatch.get(kBase + ic));
            valueCache.set(cacheOff + ic + half, vBatch.get(kBase + ic + half));
        }
    }

    /** Batched NEOX RoPE for Q only (reuse-KV layers). Worker: B*nHeads*(headDim/2). */
    public static void batchedGemmaDecodeRopeQOnly(KernelContext context,
                                                   IntArray seqPositions, FloatArray qBatch,
                                                   FloatArray freqCisReal, FloatArray freqCisImag,
                                                   int nHeads, int headDim, int qDim) {
        int half = headDim / 2;
        int g = context.globalIdx;
        int batchIdx = g / (nHeads * half);
        int rem = g % (nHeads * half);
        int h = rem / half;
        int ic = rem % half;

        int pos = seqPositions.get(batchIdx);
        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int qBase = batchIdx * qDim + h * headDim;
        float v0q = qBatch.get(qBase + ic);
        float v1q = qBatch.get(qBase + ic + half);
        qBatch.set(qBase + ic, v0q * fcr - v1q * fci);
        qBatch.set(qBase + ic + half, v0q * fci + v1q * fcr);
    }

    // ── FFN / norm sandwich (batched) ────────────────────────────────────────

    /**
     * Batched GeGLU over the packed [gate|up] GEMM output (Q8 {@code gemmMMAGateUpQ8}), emitting
     * FP16 (A operand of the W2 GEMM). {@code hb[b,i] = gelu(gate[b,i]) * up[b,i]}. Gemma uses
     * GELU (tanh approx); fork of {@code batchedFFNSwiGLUFP16Packed}. Worker: B*hiddenDim threads.
     */
    public static void batchedGemmaGeGLUPacked(KernelContext context, HalfFloatArray hbFP16,
                                               FloatArray gateUpResult, int hiddenDim) {
        int gid = context.globalIdx;
        int b = gid / hiddenDim;
        int i = gid % hiddenDim;
        int rowBase = b * 2 * hiddenDim;
        float g = gateUpResult.get(rowBase + i);
        float u = gateUpResult.get(rowBase + hiddenDim + i);
        float g3 = g * g * g;
        float gelu = 0.5f * g * (1.0f + TornadoMath.tanh(0.797885f * (g + 0.044715f * g3)));
        hbFP16.set(gid, new HalfFloat(gelu * u));
    }

    /**
     * Batched RMSNorm apply (pre-norm): {@code out[b,i] = weight[i] * (scale[b] * x[b,i])}, with
     * {@code scale[b]} from {@link org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels#batchedRmsReduceParallel}.
     * Worker: B*dim threads. Fork of {@link Gemma4Kernels#applyRmsNorm} (per-row scale).
     */
    public static void batchedGemmaApplyRmsNorm(KernelContext context, FloatArray out, FloatArray x,
                                                FloatArray weight, FloatArray scaleBatch, int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        out.set(gid, weight.get(i) * (scaleBatch.get(b) * x.get(gid)));
    }

    /** FP16-output pre-norm apply (the Q8 MMA GEMMs take a HalfFloatArray A operand). */
    public static void batchedGemmaApplyRmsNormFP16(KernelContext context, HalfFloatArray out, FloatArray x,
                                                    FloatArray weight, FloatArray scaleBatch, int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        out.set(gid, new HalfFloat(weight.get(i) * (scaleBatch.get(b) * x.get(gid))));
    }

    /**
     * Batched sandwich-norm + residual: {@code x[b,i] += weight[i] * (scale[b] * delta[b,i])}.
     * Fork of {@link Gemma4Kernels#rmsNormApplyWithResidual}. Worker: B*dim threads.
     */
    public static void batchedGemmaRmsNormApplyWithResidual(KernelContext context, FloatArray x, FloatArray delta,
                                                            FloatArray weight, FloatArray scaleBatch, int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        x.set(gid, x.get(gid) + weight.get(i) * (scaleBatch.get(b) * delta.get(gid)));
    }

    // ── Per-layer embeddings (PLE, batched) ──────────────────────────────────

    /**
     * Batched PLE gate: {@code gate[b,i] = gelu(gate[b,i]) * perLayerInputs[b, peOffset+i]}.
     * Fork of {@link Gemma4Kernels#pleGateGeluMul}; {@code size = nEmbdPerLayer},
     * {@code perLayerTotal = numLayers*nEmbdPerLayer}. Worker: B*size threads.
     */
    public static void batchedGemmaPleGateGeluMul(KernelContext context, FloatArray gate, FloatArray perLayerInputs,
                                                  int peOffset, int size, int perLayerTotal) {
        int gid = context.globalIdx;
        int b = gid / size;
        int i = gid % size;
        float g = gate.get(gid);
        float g3 = g * g * g;
        float gelu = 0.5f * g * (1.0f + TornadoMath.tanh(0.797885f * (g + 0.044715f * g3)));
        gate.set(gid, gelu * perLayerInputs.get(b * perLayerTotal + peOffset + i));
    }

    /**
     * Batched per-segment pre-scale + RMSNorm for the PLE model projection (layer-0 setup).
     * Scratch is {@code [B][numLayers][segmentSize]}; one workgroup per (slot, segment). Fork of
     * {@link Gemma4Kernels#pleProjScaleAndNormalize}. Worker: B*numLayers workgroups × localMem threads.
     */
    public static void batchedGemmaPleProjScaleAndNormalize(KernelContext context, FloatArray x, FloatArray weight,
                                                            int numLayers, int segmentSize, int localMemSize,
                                                            float preScale, float eps) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int b = groupId / numLayers;
        int seg = groupId % numLayers;
        int base = b * (numLayers * segmentSize) + seg * segmentSize;

        float[] localSum = context.allocateFloatLocalArray(64);
        float partial = 0f;
        for (int i = localId; i < segmentSize; i += localSize) {
            float v = x.get(base + i) * preScale;
            x.set(base + i, v);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = 1.0f / TornadoMath.sqrt(localSum[0] / segmentSize + eps);
        context.localBarrier();
        for (int i = localId; i < segmentSize; i += localSize) {
            x.set(base + i, weight.get(i) * (ss * x.get(base + i)));
        }
    }

    // ── Batched matvec for the mixed-precision PLE projections (thread per output) ──

    /** {@code out[b,row] = Σ_i w[row,i]·x[b,i]}, row-major F32 weight [d,n]. Worker: B*d threads. */
    public static void batchedMatVecF32(KernelContext context, FloatArray xBatch, FloatArray w,
                                        FloatArray outBatch, int n, int d) {
        int gid = context.globalIdx;
        int b = gid / d;
        int row = gid % d;
        int wBase = row * n;
        int xBase = b * n;
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += w.get(wBase + i) * xBatch.get(xBase + i);
        }
        outBatch.set(gid, acc);
    }

    /** {@code out[b,row] = Σ_i w[row,i]·x[b,i]}, row-major F16 weight [d,n], F32 accumulate. Worker: B*d threads. */
    public static void batchedMatVecF16(KernelContext context, FloatArray xBatch, HalfFloatArray w,
                                        FloatArray outBatch, int n, int d) {
        int gid = context.globalIdx;
        int b = gid / d;
        int row = gid % d;
        int wBase = row * n;
        int xBase = b * n;
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += w.get(wBase + i).getFloat32() * xBatch.get(xBase + i);
        }
        outBatch.set(gid, acc);
    }
}
