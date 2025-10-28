package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Phi3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.phi3.Phi3;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tokenizer.impl.Phi3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

public class Phi3ModelLoader extends AbstractModelLoader<Phi3, Phi3Configuration> {
    private int modelContextLength;

    public Phi3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadPhi3Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            Tokenizer tokenizer = new Phi3Tokenizer(metadata, vocabulary);
            System.out.println("Tokenizer: " + tokenizer.getClass().getSimpleName());
            return tokenizer;
        }
        return new Phi3Tokenizer(metadata, vocabulary);
    }

    @Override
    protected Phi3Configuration createConfiguration(Map<String, Object> metadata) {
        final String modelPrefix = "phi3.";
        modelContextLength = (int) metadata.get(modelPrefix + "context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;

        int vocabSize = metadata.containsKey(modelPrefix + "vocab_size") ? (int) metadata.get(modelPrefix + "vocab_size") : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new Phi3Configuration((int) metadata.get(modelPrefix + "embedding_length"),           // dim
                (int) metadata.get(modelPrefix + "feed_forward_length"),        // hidden_dim
                (int) metadata.get(modelPrefix + "block_count"),                // n_layers
                (int) metadata.get(modelPrefix + "attention.head_count"),       // n_heads

                metadata.containsKey(modelPrefix + "attention.head_count_kv") ? (int) metadata.get(modelPrefix + "attention.head_count_kv") : (int) metadata.get(modelPrefix + "attention.head_count"), // n_kv_heads

                vocabSize,                                              // vocab_size
                finalContextLength,                                                  // context_length (user-specified, not model)
                (float) metadata.getOrDefault(modelPrefix + "attention.layer_norm_rms_epsilon", 1e-5f), // rms_norm_eps
                (float) metadata.getOrDefault(modelPrefix + "rope.freq_base", 10000f)           // rope_theta
        );
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Phi3Configuration config) {
        // Calculate head size from dim and numberOfHeads
        int headSize = config.dim() / config.numberOfHeads();

        return RoPE.precomputeFreqsCis(modelContextLength,    // Use model context length for RoPE precomputation
                headSize,              // Calculated head size
                config.ropeTheta(), false,                 // Phi3 uses standard RoPE, not neox-style based on reference
                8, 1, 3, 8192         // Additional RoPE parameters from reference
        );
    }

    @Override
    protected Phi3 createModel(Phi3Configuration config, Tokenizer tokenizer, Weights weights) {
        // Phi3 chat tokens
        ChatFormat.ChatTokens chatTokens = new ChatFormat.ChatTokens("<|system|>", "<|end|>", "<|user|>", "<|end|>", "<|assistant|>");

        return new Phi3(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Phi3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
        }
        return new Phi3TornadoWeights(ModelLoader.loadTensorAsFloatArray(tokenEmbeddings), loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_norm", "weight"),
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_qkv", "weight"),      // Combined QKV
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_output", "weight"),   // wo
                loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "ffn_norm", "weight"), loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_down", "weight"),      // wDown
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_up", "weight"),        // wUp (not combined in reference)
                ModelLoader.floatBufferToFloatArray(tensorEntries.get("output_norm.weight")), FloatArray.fromArray(ropeFreqs.first()), FloatArray.fromArray(ropeFreqs.second()),
                ModelLoader.loadTensorAsHalfFloatArray(outputWeight), outputWeight.ggmlType());
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Phi3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        return new Phi3StandardWeights(ModelLoader.loadQuantized(tokenEmbeddings),                                                                               // token_embedding_table
                loadLayerWeights(tensorEntries, config, "attn_norm", "weight"),    // rms_att_weight (as FloatTensor[])
                loadLayerWeights(tensorEntries, config, "attn_qkv", "weight"),     // wqkv (combined)
                loadLayerWeights(tensorEntries, config, "attn_output", "weight"),  // wo
                loadLayerWeights(tensorEntries, config, "ffn_norm", "weight"),     // rms_ffn_weight (as FloatTensor[])
                loadLayerWeights(tensorEntries, config, "ffn_down", "weight"),     // wDown
                loadLayerWeights(tensorEntries, config, "ffn_up", "weight"),       // wUp (separate, not combined)
                ModelLoader.loadQuantized(tensorEntries.get("output_norm.weight")),                                                      // rms_final_weight (as FloatTensor)
                new ArrayFloatTensor(ropeFreqsReal),                                                                         // freq_cis_real
                new ArrayFloatTensor(ropeFreqsImag),                                                                         // freq_cis_imag
                ModelLoader.loadQuantized(outputWeight),                                                                                 // wcls
                outputWeight.ggmlType()                                                                                      // weightType
        );
    }

    // Helper methods
    private FloatTensor[] loadLayerWeights(Map<String, GGMLTensorEntry> tensorEntries, Phi3Configuration config, String layerName, String suffix) {
        FloatTensor[] weights = new FloatTensor[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadQuantized(tensorEntries.get(key));
        }
        return weights;
    }

    private FloatArray[] loadLayerWeightsAsFloatArraysFromBuffer(Map<String, GGMLTensorEntry> tensorEntries, Phi3Configuration config, String layerName, String suffix) {
        FloatArray[] weights = new FloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.floatBufferToFloatArray(tensorEntries.get(key));
        }
        return weights;
    }

    private HalfFloatArray[] loadLayerWeightsAsHalfFloatArrays(Map<String, GGMLTensorEntry> tensorEntries, Phi3Configuration config, String layerName, String suffix) {
        HalfFloatArray[] weights = new HalfFloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadTensorAsHalfFloatArray(tensorEntries.get(key));
        }
        return weights;
    }
}
