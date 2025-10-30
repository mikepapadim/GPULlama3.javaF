package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

public class Phi3TornadoVMLayerPlanner extends TornadoVMLayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeights> {

    /**
     * Constructs a TornadoVMLayerPlanner for the given Llama model.
     *
     * @param state
     *         The state object containing model tensors and buffers
     * @param model
     *         The Llama model instance containing configuration and weights
     */
    public Phi3TornadoVMLayerPlanner(Phi3State state, Model model) {
        super(state, model);
    }

    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        state.tempLogits.init(0.0f);
        final int opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());

        // @formatter:off
        TaskGraph activationUpdate = new TaskGraph("activationUpdate")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(activationUpdate.snapshot());

        TaskGraph unifiedLayer = null;
        for (int layerIndex = 0; layerIndex < config.numberOfLayers(); layerIndex++) {
            unifiedLayer = new TaskGraph("layer_" + layerIndex);
            unifiedLayer.consumeFromDevice(state.wrapX);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    weights.rms_att_weightLayered[layerIndex],
                    weights.wqkvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    weights.rms_ffn_weightLayered[layerIndex],
                    weights.wDownLayered[layerIndex],
                    weights.wUpLayered[layerIndex]
            );
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                    .task("qkvmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, state.wrapXb, state.wrapQkv,
                            weights.wqkvLayered[layerIndex], config.dim(), opSize, LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("splitQKV", TransformerComputeKernelsLayered::splitQKV,
                            state.wrapQkv, state.wrapQ, state.wrapK, state.wrapV,
                            config.dim(), config.headSize() * config.numberOfKeyValueHeads())
                    .task("rope", TransformerComputeKernelsLayered::ropeRotationPhi3,context,
                            state.positionHolder, state.wrapQ, state.wrapK, config.kvDim(),
                            config.headSize())
                    .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                            state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength())
                    .task("parallel-attention", TransformerComputeKernelsLayered::processHeadsFlashAttention, context,
                            state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                            config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                            state.positionHolder, layerIndex, config.contextLength())
                    .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapXb,  state.wrapX, weights.woLayered[layerIndex], config.dim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                    .task("wGateUp", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb,   state.wrapHb, weights.wUpLayered[layerIndex],  config.dim(), 2 * config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("gateUpSiLU", TransformerComputeKernelsLayered::splitGateUpAndSiLU,
                            state.wrapHb, state.wrapHbG, state.wrapHbU, config.hiddenDim())
                    .task("wDown", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapHbU, state.wrapX, weights.wDownLayered[layerIndex], config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
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

        return new Tuple2<>(taskGraphs, setupGridSchedulersLayered());
    }

    // @formatter:off
    /**
     * Configures the final projection layer in the task graph based on weight quantization type.
     *
     * This method adds a "projection" task to compute the final logits by performing a
     * matrix-vector multiplication between the model's output embeddings and the classifier
     * weights (wcls). The computation kernel used depends on the quantization format.
     *
     * Supported quantization types:
     * - Q8_0: 8-bit quantization with uniform scaling per 32-element block
     * - Q4_0: 4-bit quantization with uniform scaling per 32-element block
     *
     * The task multiplies:
     * - weights.wclsByteArray: Quantized classifier weights (vocab_size x dim)
     * - state.wrapX: Current layer output (dim)
     * - Result: state.wrapLogits: Raw logits (vocab_size)
     *
     * @param logits The existing task graph to extend with the projection operation
     * @return The modified task graph with the projection task added
     * @throws UnsupportedOperationException If weights.weightType is not Q8_0 or Q4_0
     */
    // @formatter:on
    protected TaskGraph configureQuantizedMatrixVectorFinalWeight(TaskGraph logits) {
        switch (weights.getWeightType()) {
            case F16:
            case Q8_0:
            case Q4_0:
                logits.task("projection", TransformerComputeKernelsLayered::matrixVectorGeneric,  //
                        context, state.wrapX, state.wrapLogits, weights.wclsHalfFloat, //
                        config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS); //
                break;
            default:
                throw new UnsupportedOperationException("Unsupported weight quantization type: " + weights.getWeightType() + ". Only Q8_0 and Q4_0 are supported.");
        }
        return logits;
    }

    /**
     * Configures data transfer operations for a specific layer in the neural network task graph.
     *
     * This method manages GPU memory transfers with optimized data movement strategies: This optimization pattern minimizes data movement by: 1. Using one-time transfers for static data 2. Reusing
     * intermediate results already on GPU from previous layers 3. Only transferring // dynamic data that changes per execution
     *
     * @param unifiedLayer
     *         The task graph representing this layer's operations
     * @param layerIndex
     *         Index of the current layer (0-based)
     * @return The configured task graph with appropriate data transfer operations
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionHolder, state.temp, state.tempFFN); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.wrapHbG, state.wrapHbU, state.wrapQkv); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder, // /
                    state.wrapHbG, state.wrapHbU, state.wrapQkv);
        }
        return unifiedLayer;
    }

    // @formatter:off
    /**
     * Sets up the grid scheduler configuration for a layered neural network forward pass.
     *
     * This method creates and configures worker grids for different types of GPU operations
     * in the transformer/ML model pipeline. Each worker grid defines how work should be
     * distributed across GPU threads (OpenCL work-items or CUDA threads).
     *
     * The method creates several worker profiles:
     * - Single thread operations (activation updates)
     * - RoPE (Rotary Position Embedding) operations
     * - Matrix multiplications with different dimensions
     * - RMS normalization operations
     * - Parallel attention computations
     * - Cache copying operations
     * - Vocabulary projections
     *
     * Each worker grid maps to equivalent OpenCL NDRange or CUDA grid/block configurations:
     * - setGlobalWork() ≈ OpenCL global_work_size ≈ CUDA grid dimensions × block dimensions
     * - setLocalWork() ≈ OpenCL local_work_size ≈ CUDA block dimensions
     *
     * @return GridScheduler configured with all necessary worker grids for the model layers
     */
    // @formatter:on
    private GridScheduler setupGridSchedulersLayered() {
        // Phi3-specific: Combined QKV dimension (opSize = dim + 2 * kvDim)
        final int opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());
        WorkerGrid qkvmatmulWorker = GridSchedulerBuilder.createWorker1D(
                opSize * LOCAL_WORK_GROUP_SIZE_ALLOC,
                LOCAL_WORK_GROUP_SIZE_ALLOC
        );

        // Phi3-specific: Split QKV worker
        WorkerGrid splitQKVWorker = GridSchedulerBuilder.createWorker1D(opSize, 128);

        // Phi3-specific: Gate/Up combined worker (2 * hiddenDim)
        int wGateUpGlobal = 2 * config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid wGateUpWorker = GridSchedulerBuilder.createWorker1D(wGateUpGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Phi3-specific: Split gate/up with SiLU activation
        WorkerGrid splitGateUpSiLUWorker = GridSchedulerBuilder.createWorker1D(config.hiddenDim(), 128);

        // Build scheduler with common workers
        GridScheduler scheduler = new GridSchedulerBuilder(config, 256)
                .addSingleWorker()
                .addRopeWorker()
                .addConfigDimWorker()  // Used for matmul1 and wDown
                .addRmsNormWorker()
                .addParallelAttentionWorker(8)  // 8 threads per head for Phi3
                .addVocabWorker()
                .build();

        // Phi3-specific: Custom copy worker (kvDim dimension, global work is dim)
        WorkerGrid copyToCachesWorker = GridSchedulerBuilder.createWorker1D(config.kvDim(), 128);
        // Override global work to dim instead of kvDim
        copyToCachesWorker.setGlobalWork(config.dim(), 1, 1);

        // Register Phi3-specific workers for all layers
        for (int i = 0; i < config.numberOfLayers(); i++) {
            scheduler.addWorkerGrid("layer_" + i + ".qkvmatmul", qkvmatmulWorker);
            scheduler.addWorkerGrid("layer_" + i + ".splitQKV", splitQKVWorker);
            scheduler.addWorkerGrid("layer_" + i + ".wGateUp", wGateUpWorker);
            scheduler.addWorkerGrid("layer_" + i + ".gateUpSiLU", splitGateUpSiLUWorker);
            scheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
        }

        return scheduler;
    }
}
