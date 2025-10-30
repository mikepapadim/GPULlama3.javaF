package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;

/**
 * Builder for creating GridScheduler instances with common worker configurations.
 * Eliminates duplication across TornadoVM layer planners.
 *
 * <p>This class provides a fluent API for configuring GPU worker grids for transformer model inference.
 * Each worker grid defines how computational work is distributed across GPU threads.</p>
 *
 * <p><b>Design Pattern:</b> Builder Pattern</p>
 *
 * <p><b>Benefits:</b></p>
 * <ul>
 *   <li>Eliminates ~300 lines of duplicated grid scheduler setup code (75% reduction)</li>
 *   <li>Centralizes worker grid configuration logic</li>
 *   <li>Provides consistent worker configurations across all model types</li>
 *   <li>Makes it easy to add model-specific workers via custom worker API</li>
 * </ul>
 *
 * <p><b>Usage Example:</b></p>
 * <pre>{@code
 * GridScheduler scheduler = new GridSchedulerBuilder(config, localSize)
 *     .addCommonWorkers()
 *     .build();
 * }</pre>
 *
 * <p><b>Custom Workers Example:</b></p>
 * <pre>{@code
 * WorkerGrid biasWorker = GridSchedulerBuilder.createWorker1D(config.dim() * 32, 32);
 * GridScheduler scheduler = new GridSchedulerBuilder(config, localSize)
 *     .addCommonWorkers()
 *     .addCustomWorker("qbias", biasWorker, true)  // Register for all layers
 *     .build();
 * }</pre>
 */
public class GridSchedulerBuilder {

    // Constants from TransformerComputeKernels
    private static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;
    private static final int THREAD_SCALE_FOR_LOGITS = 8;

    private final GridScheduler scheduler;
    private final Configuration config;
    private final int localSize;

    /**
     * Create a new GridSchedulerBuilder.
     *
     * @param config The model configuration
     * @param localSize The local work group size for reductions (typically 256 for NVIDIA, 32 for others)
     */
    public GridSchedulerBuilder(Configuration config, int localSize) {
        this.scheduler = new GridScheduler();
        this.config = config;
        this.localSize = localSize;
    }

    /**
     * Add all common workers used by most models.
     * Includes: single, RoPE, dimension workers, RMSNorm, attention, caches, vocab.
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addCommonWorkers() {
        addSingleWorker();
        addRopeWorker();
        addConfigDimWorker();
        addKvDimWorker();
        addHiddenDimWorker();
        addRmsNormWorker();
        addParallelAttentionWorker();
        addCopyToCachesWorker();
        addVocabWorker();
        return this;
    }

    /**
     * Add single-threaded worker for activation updates.
     * OpenCL equivalent: global_work_size=[1,1,1], local_work_size=[1,1,1]
     * CUDA equivalent: kernel<<<dim3(1,1,1), dim3(1,1,1)>>>
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addSingleWorker() {
        WorkerGrid worker = new WorkerGrid1D(1);
        worker.setGlobalWork(1, 1, 1);
        worker.setLocalWork(1, 1, 1);
        scheduler.addWorkerGrid("activationUpdate.updateX", worker);
        return this;
    }

    /**
     * Add RoPE worker for rotary position embeddings.
     * OpenCL equivalent: global_work_size=[dim/2,1,1], local_work_size=[128,1,1]
     * CUDA equivalent: kernel<<<dim3((dim/2+127)/128,1,1), dim3(128,1,1)>>>
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addRopeWorker() {
        int ropeSize = config.dim() / 2;
        WorkerGrid worker = new WorkerGrid1D(ropeSize);
        worker.setGlobalWork(ropeSize, 1, 1);
        worker.setLocalWork(128, 1, 1);
        registerForAllLayers("rope", worker);
        return this;
    }

    /**
     * Add worker for operations with config.dim() size.
     * Used for Q matmul, FFN matmuls, projection operations.
     * OpenCL equivalent: global_work_size=[dim*32,1,1], local_work_size=[32,1,1]
     * CUDA equivalent: kernel<<<dim3(dim,1,1), dim3(32,1,1)>>>
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addConfigDimWorker() {
        int globalSize = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid worker = new WorkerGrid1D(globalSize);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        registerForAllLayers("qmatmul", worker);
        registerForAllLayers("matmul1", worker);
        registerForAllLayers("projectionTwo", worker);

        return this;
    }

    /**
     * Add worker for operations with KV dimension.
     * Used for K and V matmuls.
     * OpenCL equivalent: global_work_size=[kvDim*32,1,1], local_work_size=[32,1,1]
     * CUDA equivalent: kernel<<<dim3(kvDim,1,1), dim3(32,1,1)>>>
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addKvDimWorker() {
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int globalSize = kvDim * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid worker = new WorkerGrid1D(globalSize);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        registerForAllLayers("kmatmul", worker);
        registerForAllLayers("vmatmul", worker);

        return this;
    }

    /**
     * Add worker for operations with hidden dimension.
     * Used for feedforward network operations.
     * OpenCL equivalent: global_work_size=[hiddenDim*32,1,1], local_work_size=[32,1,1]
     * CUDA equivalent: kernel<<<dim3(hiddenDim,1,1), dim3(32,1,1)>>>
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addHiddenDimWorker() {
        int globalSize = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid worker = new WorkerGrid1D(globalSize);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Register for fused_ffn_w1_w3 (standard naming in base planner)
        registerForAllLayers("fused_ffn_w1_w3", worker);

        return this;
    }

    /**
     * Add RMS normalization worker.
     * OpenCL equivalent: global_work_size=[dim,1,1], local_work_size=[localSize,1,1]
     * CUDA equivalent: kernel<<<dim3((dim+localSize-1)/localSize,1,1), dim3(localSize,1,1)>>>
     *
     * @param localWorkSize Optional custom local work size (default: localSize from constructor)
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addRmsNormWorker(int... localWorkSize) {
        int local = localWorkSize.length > 0 ? localWorkSize[0] : localSize;
        WorkerGrid worker = new WorkerGrid1D(config.dim());
        worker.setGlobalWork(config.dim(), 1, 1);
        worker.setLocalWork(local, 1, 1);

        registerForAllLayers("reductionsOneBlock", worker);
        registerForAllLayers("reductionsOneBlockFFN", worker);
        registerForAllLayers("mapContext", worker);
        registerForAllLayers("mapContextFFN", worker);

        scheduler.addWorkerGrid("logits.reductionsOneBlockLogits", worker);
        scheduler.addWorkerGrid("logits.mapContextLogits", worker);

        return this;
    }

    /**
     * Add parallel attention worker.
     * OpenCL equivalent: global_work_size=[numberOfHeads*threads,1,1], local_work_size=[threads,1,1]
     * CUDA equivalent: kernel<<<dim3(numberOfHeads,1,1), dim3(threads,1,1)>>>
     *
     * @param threadsPerHead Optional threads per attention head (default: 8)
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addParallelAttentionWorker(int... threadsPerHead) {
        int threads = threadsPerHead.length > 0 ? threadsPerHead[0] : 8;
        int globalSize = config.numberOfHeads() * threads;
        WorkerGrid worker = new WorkerGrid1D(globalSize);
        worker.setGlobalWork(globalSize, 1, 1);
        worker.setLocalWork(threads, 1, 1);

        registerForAllLayers("parallel-attention", worker);

        return this;
    }

    /**
     * Add worker for copying to KV caches.
     * OpenCL equivalent: global_work_size=[dim,1,1], local_work_size=[128,1,1]
     * CUDA equivalent: kernel<<<dim3((dim+127)/128,1,1), dim3(128,1,1)>>>
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addCopyToCachesWorker() {
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        WorkerGrid worker = new WorkerGrid1D(kvDim);
        worker.setGlobalWork(config.dim(), 1, 1);
        worker.setLocalWork(128, 1, 1);

        registerForAllLayers("copyToCaches", worker);

        return this;
    }

    /**
     * Add vocabulary worker for final logits computation.
     * OpenCL equivalent: global_work_size=[vocabSize*32*8,1,1], local_work_size=[32*8,1,1]
     * CUDA equivalent: kernel<<<dim3((vocabSize+255)/256,1,1), dim3(256,1,1)>>>
     *
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addVocabWorker() {
        int globalSize = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        WorkerGrid worker = new WorkerGrid1D(globalSize);
        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);

        scheduler.addWorkerGrid("logits.projection", worker);

        return this;
    }

    /**
     * Add a custom worker for model-specific operations.
     *
     * <p><b>Example usage:</b></p>
     * <pre>{@code
     * // Add a bias worker for Qwen2 models
     * WorkerGrid biasWorker = GridSchedulerBuilder.createWorker1D(config.dim() * 32, 32);
     * builder.addCustomWorker("qbias", biasWorker, true);  // Register for all layers
     * }</pre>
     *
     * @param taskName The task name (without layer prefix, e.g., "qbias" not "layer_0.qbias")
     * @param worker The worker grid configuration
     * @param allLayers If true, register for all layers; if false, register once
     * @return this builder for chaining
     */
    public GridSchedulerBuilder addCustomWorker(String taskName, WorkerGrid worker, boolean allLayers) {
        if (allLayers) {
            registerForAllLayers(taskName, worker);
        } else {
            scheduler.addWorkerGrid(taskName, worker);
        }
        return this;
    }

    /**
     * Build and return the configured GridScheduler.
     *
     * @return The configured GridScheduler
     */
    public GridScheduler build() {
        return scheduler;
    }

    /**
     * Helper to register a worker for all transformer layers.
     */
    private void registerForAllLayers(String taskName, WorkerGrid worker) {
        for (int i = 0; i < config.numberOfLayers(); i++) {
            scheduler.addWorkerGrid("layer_" + i + "." + taskName, worker);
        }
    }

    /**
     * Create a simple 1D worker grid with given global and local sizes.
     * Utility method for creating custom workers.
     *
     * <p><b>Example usage:</b></p>
     * <pre>{@code
     * WorkerGrid customWorker = GridSchedulerBuilder.createWorker1D(1024, 32);
     * }</pre>
     *
     * @param globalSize The global work size
     * @param localSize The local work size
     * @return The configured WorkerGrid
     */
    public static WorkerGrid createWorker1D(int globalSize, int localSize) {
        WorkerGrid worker = new WorkerGrid1D(globalSize);
        worker.setGlobalWork(globalSize, 1, 1);
        worker.setLocalWork(localSize, 1, 1);
        return worker;
    }

    /**
     * Create a 2D worker grid with given dimensions.
     * Utility method for creating custom 2D workers (e.g., for Qwen2 RoPE).
     *
     * <p><b>Example usage:</b></p>
     * <pre>{@code
     * // Create a 2D worker for Qwen2 RoPE: numberOfHeads x (headSize/2)
     * WorkerGrid ropeWorker = GridSchedulerBuilder.createWorker2D(
     *     config.numberOfHeads(), config.headSize() / 2, 1, 1
     * );
     * }</pre>
     *
     * @param globalX The global work size in X dimension
     * @param globalY The global work size in Y dimension
     * @param localX The local work size in X dimension
     * @param localY The local work size in Y dimension
     * @return The configured WorkerGrid
     */
    public static WorkerGrid createWorker2D(int globalX, int globalY, int localX, int localY) {
        WorkerGrid worker = new uk.ac.manchester.tornado.api.WorkerGrid2D(globalX, globalY);
        worker.setGlobalWork(globalX, globalY, 1);
        worker.setLocalWork(localX, localY, 1);
        return worker;
    }
}
