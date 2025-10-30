package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

public class Qwen3TornadoVMLayerPlanner extends TornadoVMLayerPlanner<Qwen3State, Qwen3Configuration, Qwen3TornadoWeights> {

    private final int nHeadKv;
    private final int nEmbdHeadK;
    private final int nEmbdHeadV;
    private final int nEmbdVGqa;
    private final int nEmbdHead;
    private final int nEmbdGqa;
    private final int gqa;

    public Qwen3TornadoVMLayerPlanner(Qwen3State state, Model model) {
        super(state, model);

        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue(); // n_embd_head_v = n_embd / n_head; %s.attention.value_length
        this.nEmbdVGqa = nEmbdHeadV * nHeadKv; // n_embd_v_gqa = n_embd_head_v * n_head_kv
        this.nEmbdHead = nEmbdHeadV;
        this.nEmbdGqa = nEmbdVGqa;
        this.gqa = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
    }

    // @formatter:off
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN,
                    state.tempQcur, state.tempKcur); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb);//
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder //
            );
        }
        return unifiedLayer;
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        state.tempLogits.init(0.0f);
        state.wrapLogits.init(0.0f);

        TaskGraph activationUpdate = new TaskGraph("activationUpdate")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(activationUpdate.snapshot());

        TaskGraph unifiedLayer = null;
        for (int layerIndex =0; layerIndex < config.numberOfLayers(); layerIndex++) {
            unifiedLayer = new TaskGraph("layer_" + layerIndex);
            unifiedLayer.consumeFromDevice(state.wrapX);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    //Copy-in weights per layer for batched-layered layout
                    weights.rms_att_weightLayered[layerIndex],
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    //rms_att_KNormLayered
                    weights.rms_att_KNormLayered[layerIndex],
                    //rms_att_QNormLayered
                    weights.rms_att_QNormLayered[layerIndex],
                    weights.rms_ffn_weightLayered[layerIndex],
                    weights.w1Layered[layerIndex],
                    weights.w2Layered[layerIndex],
                    weights.w3Layered[layerIndex]
            );
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            unifiedLayer.task("reductionsOneBlock",
                                    TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                                    context,
                                    state.temp,
                                    state.wrapX, // in
                                    config.dim(),
                                    config.rmsNormEps(),
                                    state.localSize)
                            .task("mapContext",
                                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                                    context,
                                    state.wrapXb, // out
                                    state.wrapX,
                                    weights.rms_att_weightLayered[layerIndex],
                                    state.temp);

            int qDim0 = nEmbdHeadK * config.numberOfHeads();
            int kvDim0 = nEmbdGqa;
            int qkvDim1 = config.dim();
            unifiedLayer.task("qmatmul",
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            state.wrapXb,
                            state.wrapQ,                    // output
                            weights.wqLayered[layerIndex],
                            qkvDim1,
                            qDim0,
                            LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("kmatmul",
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            state.wrapXb,
                            state.wrapK,        // output
                            weights.wkLayered[layerIndex],
                            qkvDim1,
                            kvDim0,
                            LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("vmatmul",
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            state.wrapXb,
                            state.wrapV,        // output
                            weights.wvLayered[layerIndex],
                            qkvDim1,
                            kvDim0,
                            LOCAL_WORK_GROUP_SIZE_ALLOC);

            // Qcur rmsnorm
            unifiedLayer
                    .task("rmsnormReduction_Qcur",
                            Qwen3Kernels::rmsnormWithParallelOffset,
                            context,
                            state.tempQcur,         // output
                            state.wrapQ,            // input
                            state.localSize,        // currently 128, should be variable of global nEmbHead
                            nEmbdHead,              // for normalization
                            config.rmsNormEps())    // for normalization
                    .task("rmsnormMapIndexInPlace_Qcur",
                            Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset,
                            context,
                            state.wrapQ,        // output
                            weights.rms_att_QNormLayered[layerIndex],
                            nEmbdHead,
                            state.tempQcur);

            // Kcur rmsnorm
            unifiedLayer
                    .task("rmsnormReduction_Kcur",
                            Qwen3Kernels::rmsnormWithParallelOffset,
                            context,
                            state.tempKcur,         // output
                            state.wrapK,            // input
                            state.localSize,        // currently 128, should be variable of global nEmbHead
                            nEmbdHead,              // for normalization
                            config.rmsNormEps())    // for normalization
                    .task("rmsnormMapIndexInPlace_Kcur",
                            Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset,
                            context,
                            state.wrapK,        // output
                            weights.rms_att_KNormLayered[layerIndex],
                            nEmbdHead,
                            state.tempKcur);

            // rope rotation task graph
            unifiedLayer.task("ropeRotation",
                            Qwen3Kernels::ropeRotation,
                            context,
                            state.positionHolder,
                            state.wrapQ,            // out
                            state.wrapK,            // out
                            config.numberOfKeyValueHeads(),
                            nEmbdHead);

            unifiedLayer.task("copyToCaches",
                    TransformerComputeKernelsLayered::copyToCache,
                    state.wrapKeyCache,         // out
                    state.wrapK,                // in
                    state.wrapValueCache,       // out
                    state.wrapV,                // in
                    state.positionHolder,
                    nEmbdGqa,
                    layerIndex,
                    config.contextLength());

            unifiedLayer.task("parallel-attention",
                    TransformerComputeKernelsLayered::processHeadsFlashAttentionOpt,
                    context,
                    state.wrapQ,
                    state.wrapKeyCache,
                    state.wrapValueCache,
                    state.wrapXb,               // out
                    config.numberOfHeads(),
                    nEmbdHead,
                    nEmbdGqa,
                    gqa,
                    state.positionHolder,
                    layerIndex,
                    config.contextLength());

            unifiedLayer.task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                    context,
                    state.wrapXb,                           // vector
                    state.wrapX,                            // out, should be [1024]
                    weights.woLayered[layerIndex],          // matrix
                    nEmbdHeadK * config.numberOfHeads(),    // dim1 = 2048
                    config.dim(),                           // dim0 = 1024
                    LOCAL_WORK_GROUP_SIZE_ALLOC);

            unifiedLayer.task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                            context, state.tempFFN, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("reductionFinalNormalizationFFN" , TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempFFN,
                            config.dim(), config.rmsNormEps())
                    .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN);

            unifiedLayer.task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                            state.wrapXb,   state.wrapHb, weights.w1Layered[layerIndex], weights.w3Layered[layerIndex], config.dim(), config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapHb, state.wrapX, weights.w2Layered[layerIndex], config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .persistOnDevice(
                            state.wrapX
                    );
            taskGraphs.add(unifiedLayer.snapshot());
        }

        TaskGraph lastUnifiedLayer = unifiedLayer;
        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(lastUnifiedLayer.getTaskGraphName(),
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits,
                        state.wrapLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer,
                        context,
                        state.tempLogits,
                        state.wrapX,
                        config.dim(),
                        config.rmsNormEps(),
                        state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);
        logits = configureQuantizedMatrixVectorFinalWeight(logits);
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());

        return new Tuple2<>(taskGraphs, setupQwen3GridSchedulersLayeredNonNvidia());

    }
    // @formatter:on

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
        return setupTornadoForwardPlanLayered();
    }

    private GridScheduler setupQwen3GridSchedulersLayeredNonNvidia() {
        // Qwen3-specific: Custom Q matmul worker (nEmbdHeadK * numberOfHeads)
        int matmulQGlobal = nEmbdHeadK * config.numberOfHeads() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = GridSchedulerBuilder.createWorker1D(matmulQGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Qwen3-specific: Custom K/V matmul worker (nEmbdGqa)
        int matmulKVGlobal = nEmbdGqa * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = GridSchedulerBuilder.createWorker1D(matmulKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Qwen3-specific: Q/K normalization workers
        WorkerGrid qCurWorker = GridSchedulerBuilder.createWorker1D(config.numberOfHeads() * nEmbdHead, nEmbdHead);
        WorkerGrid kCurWorker = GridSchedulerBuilder.createWorker1D(config.numberOfKeyValueHeads() * nEmbdHead, nEmbdHead);

        // Qwen3-specific: 2D RoPE worker (numberOfHeads x nEmbdHead/2)
        WorkerGrid ropeWorker = GridSchedulerBuilder.createWorker2D(
                config.numberOfHeads(),
                nEmbdHead / 2,
                8, 1
        );

        // Qwen3-specific: Custom copy worker using nEmbdGqa dimension
        WorkerGrid copyToCachesWorker = GridSchedulerBuilder.createWorker1D(nEmbdGqa, 128);

        // Qwen3-specific: Parallel attention with 32 threads per head
        WorkerGrid parallelAttentionWorker = GridSchedulerBuilder.createWorker1D(
                config.numberOfHeads() * 32,
                32
        );

        // Build scheduler with common workers and override with Qwen3-specific ones
        GridScheduler scheduler = new GridSchedulerBuilder(config, state.localSize)
                .addSingleWorker()
                .addRmsNormWorker()
                .addHiddenDimWorker()
                .addVocabWorker()
                .build();

        // Register Qwen3-specific workers for all layers
        for (int i = 0; i < config.numberOfLayers(); i++) {
            scheduler.addWorkerGrid("layer_" + i + ".qmatmul", matmulQRowMajorWorker);
            scheduler.addWorkerGrid("layer_" + i + ".kmatmul", matmulKVRowMajorWorker);
            scheduler.addWorkerGrid("layer_" + i + ".vmatmul", matmulKVRowMajorWorker);
            scheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Qcur", qCurWorker);
            scheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Qcur", qCurWorker);
            scheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Kcur", kCurWorker);
            scheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Kcur", kCurWorker);
            scheduler.addWorkerGrid("layer_" + i + ".ropeRotation", ropeWorker);
            scheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            scheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);

            // Standard workers
            int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid matmul1Worker = GridSchedulerBuilder.createWorker1D(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);
            scheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);

            // FFN workers (shared with standard dimensions)
            int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid projectionTwoWorker = GridSchedulerBuilder.createWorker1D(projectionTwoGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);
            scheduler.addWorkerGrid("layer_" + i + ".projectionTwo", projectionTwoWorker);
        }

        return scheduler;
    }

}
