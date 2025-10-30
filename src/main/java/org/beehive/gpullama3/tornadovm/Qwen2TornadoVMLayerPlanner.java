package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

public class Qwen2TornadoVMLayerPlanner extends TornadoVMLayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights> {

    /**
     * Constructs a TornadoVMLayerPlanner for the given Qwen2 model.
     *
     * @param state
     *         The state object containing model tensors and buffers
     * @param model
     *         The Qwen2 model instance containing configuration and weights
     */
    public Qwen2TornadoVMLayerPlanner(Qwen2State state, Model model) {
        super(state, model);
    }

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
                    weights.q_biasLayered[layerIndex],
                    weights.k_biasLayered[layerIndex],
                    weights.v_biasLayered[layerIndex],
                    weights.rms_ffn_weightLayered[layerIndex],
                    weights.w1Layered[layerIndex],
                    weights.w2Layered[layerIndex],
                    weights.w3Layered[layerIndex]
            );
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

            unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                    .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb,  state.wrapQ, weights.wqLayered[layerIndex], config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb,  state.wrapK, weights.wkLayered[layerIndex], config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb,   state.wrapV, weights.wvLayered[layerIndex], config.dim(), config.kvDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("qbias", TransformerComputeKernelsLayered::addInPlace, state.wrapQ, weights.q_biasLayered[layerIndex], config.dim())
                    .task("kbias", TransformerComputeKernelsLayered::addInPlace, state.wrapK, weights.k_biasLayered[layerIndex], config.kvDim())
                    .task("vbias", TransformerComputeKernelsLayered::addInPlace, state.wrapV, weights.v_biasLayered[layerIndex], config.kvDim())
                    .task("rope", Qwen3Kernels::ropeRotation,context, state.positionHolder, state.wrapQ, state.wrapK, config.numberOfKeyValueHeads(),
                            config.headSize())
                    .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                            state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength())
                    .task("parallel-attention", Qwen2Kernels::processHeadsFlashAttention, context,
                            state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                            config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                            state.positionHolder, layerIndex, config.contextLength())
                    .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapXb,  state.wrapX, weights.woLayered[layerIndex], config.dim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                    .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
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
                        state.tempLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapLogits,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);
        logits = configureQuantizedMatrixVectorFinalWeight(logits);
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());
        // @formatter:on

        return new Tuple2<>(taskGraphs, setupQwen2GridSchedulersLayeredNonNvidia());
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
        return setupTornadoForwardPlanLayered();
    }

    /**
     * Sets up the grid scheduler configuration for Qwen2 model on non-NVIDIA GPUs.
     * Qwen2 has additional bias workers (qbias, kbias, vbias) and uses a 2D RoPE worker.
     *
     * @return GridScheduler configured with Qwen2-specific worker grids
     */
    private GridScheduler setupQwen2GridSchedulersLayeredNonNvidia() {
        // Qwen2-specific: 2D RoPE worker (numberOfHeads x headSize/2)
        WorkerGrid ropeWorker2D = GridSchedulerBuilder.createWorker2D(
                config.numberOfHeads(),
                config.headSize() / 2,
                1, 1
        );

        // Qwen2-specific: Q bias worker
        WorkerGrid qBiasWorker = GridSchedulerBuilder.createWorker1D(
                config.dim(),
                config.dim() / 8
        );

        // Qwen2-specific: K/V bias worker
        WorkerGrid kvBiasWorker = GridSchedulerBuilder.createWorker1D(
                config.kvDim(),
                32
        );

        // Qwen2-specific: Optimal attention worker (find best divisor)
        int optimalLocalSize = Math.min(config.headSize(), 64);
        if (config.headSize() % optimalLocalSize != 0) {
            for (int size = 64; size >= 1; size--) {
                if (config.headSize() % size == 0) {
                    optimalLocalSize = size;
                    break;
                }
            }
        }
        WorkerGrid attentionWorker = GridSchedulerBuilder.createWorker1D(
                config.numberOfHeads() * optimalLocalSize,
                optimalLocalSize
        );

        // Qwen2-specific: Custom copy worker with local size 32
        WorkerGrid copyWorker = GridSchedulerBuilder.createWorker1D(
                config.kvDim(),
                32
        );

        // Build scheduler with common workers (using local size 32 for RMS norm)
        // and override specific workers for Qwen2
        GridScheduler scheduler = new GridScheduler();

        // Add single worker
        new GridSchedulerBuilder(config, 32)
                .addSingleWorker();

        // Manually add all workers since we need custom versions
        WorkerGrid singleWorker = GridSchedulerBuilder.createWorker1D(1, 1);
        scheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);

        // Add dimension workers
        int configDimGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimWorker = GridSchedulerBuilder.createWorker1D(configDimGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configKvDimGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimWorker = GridSchedulerBuilder.createWorker1D(configKvDimGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimGlobal = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimWorker = GridSchedulerBuilder.createWorker1D(configHiddenDimGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // RMS norm worker with local size 32
        WorkerGrid rmsNormWorker = GridSchedulerBuilder.createWorker1D(config.dim(), 32);

        // Register for all layers
        for (int i = 0; i < config.numberOfLayers(); i++) {
            scheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimWorker);
            scheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimWorker);
            scheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimWorker);
            scheduler.addWorkerGrid("layer_" + i + ".qbias", qBiasWorker);
            scheduler.addWorkerGrid("layer_" + i + ".kbias", kvBiasWorker);
            scheduler.addWorkerGrid("layer_" + i + ".vbias", kvBiasWorker);
            scheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker2D);  // Custom 2D worker
            scheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimWorker);
            scheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimWorker);
            scheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimWorker);
            scheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            scheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            scheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            scheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            scheduler.addWorkerGrid("layer_" + i + ".parallel-attention", attentionWorker);  // Custom optimal worker
            scheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyWorker);  // Custom copy worker
        }

        // Vocabulary worker
        int vocabGlobal = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        WorkerGrid vocabWorker = GridSchedulerBuilder.createWorker1D(vocabGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);

        scheduler.addWorkerGrid("logits.projection", vocabWorker);
        scheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
        scheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

        return scheduler;
    }
}
