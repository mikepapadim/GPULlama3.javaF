package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.qwen2.Qwen2;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tokenizer.impl.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadQwen3Vocabulary;

public class Qwen2ModelLoader extends AbstractModelLoader<Qwen2, Qwen2Configuration> {

    public Qwen2ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return loadQwen3Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        return new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);
    }

    @Override
    protected Qwen2Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen2.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;

        int numberOfKeyValueHeads = metadata.containsKey("qwen2.attention.head_count_kv") ? (int) metadata.get("qwen2.attention.head_count_kv") : (int) metadata.get("qwen2.attention.head_count");
        int vocabSize = metadata.containsKey("qwen2.vocab_size") ? (int) metadata.get("qwen2.vocab_size") : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new Qwen2Configuration((int) metadata.get("qwen2.embedding_length"),       // dim
                (int) metadata.get("qwen2.feed_forward_length"),    // hiddendim
                (int) metadata.get("qwen2.block_count"),            // numberOfLayers
                (int) metadata.get("qwen2.attention.head_count"),   // numberOfHeads

                numberOfKeyValueHeads, // numberOfKeyValueHeads
                numberOfKeyValueHeads, // numberOfHeadsKey
                numberOfKeyValueHeads, // numberOfHeadsValue

                vocabSize, modelContextLength, finalContextLength, false, (float) metadata.get("qwen2.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("qwen2.rope.freq_base"));
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen2Configuration config) {
        return RoPE.precomputeFreqsCis(config.contextLengthModel(), config.headSize(), config.ropeTheta(), false, 8, 1, 3, 8192);
    }

    @Override
    protected Qwen2 createModel(Qwen2Configuration config, Tokenizer tokenizer, Weights weights) {
        Map<String, Object> metadata = gguf.getMetadata();
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        // Qwen2.5-Coder uses <|endoftext|> as stop-token.
        ChatTokens chatTokens = isDeepSeekR1DistillQwen ? new ChatTokens("<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "")
                : new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen2(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Qwen2StandardWeights(ModelLoader.loadQuantized(tokenEmbeddings), loadLayerWeights(tensorEntries, config, "attn_norm", "weight"),
                loadLayerWeights(tensorEntries, config, "attn_q", "weight"), loadLayerWeights(tensorEntries, config, "attn_k", "weight"), loadLayerWeights(tensorEntries, config, "attn_v", "weight"),

                loadLayerWeights(tensorEntries, config, "attn_q", "bias"), loadLayerWeights(tensorEntries, config, "attn_k", "bias"), loadLayerWeights(tensorEntries, config, "attn_v", "bias"),

                loadLayerWeights(tensorEntries, config, "attn_output", "weight"), loadLayerWeights(tensorEntries, config, "ffn_norm", "weight"),
                loadLayerWeights(tensorEntries, config, "ffn_gate", "weight"), loadLayerWeights(tensorEntries, config, "ffn_down", "weight"), loadLayerWeights(tensorEntries, config, "ffn_up", "weight"),
                ModelLoader.loadQuantized(tensorEntries.get("output_norm.weight")), new ArrayFloatTensor(ropeFreqs.first()), new ArrayFloatTensor(ropeFreqs.second()),
                ModelLoader.loadQuantized(outputWeight), outputWeight.ggmlType());
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
        }
        return new Qwen2TornadoWeights(ModelLoader.loadTensorAsFloatArray(tokenEmbeddings),
                loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_norm", "weight"), loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_q", "weight"),
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_k", "weight"), loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_v", "weight"),
                // Qwen2-specific: qkv bias
                loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_q", "bias"), loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_k", "bias"),
                loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_v", "bias"),

                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_output", "weight"), loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "ffn_norm", "weight"),
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_gate", "weight"),            // w1
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_down", "weight"),            // w2
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_up", "weight"),              // w3
                ModelLoader.floatBufferToFloatArray(tensorEntries.get("output_norm.weight")), FloatArray.fromArray(ropeFreqs.first()), FloatArray.fromArray(ropeFreqs.second()),
                ModelLoader.loadTensorAsHalfFloatArray(outputWeight), outputWeight.ggmlType());
    }

    // Helper methods
    private FloatTensor[] loadLayerWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, String layerName, String suffix) {
        FloatTensor[] weights = new FloatTensor[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadQuantized(tensorEntries.get(key));
        }
        return weights;
    }

    private FloatArray[] loadLayerWeightsAsFloatArraysFromBuffer(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, String layerName, String suffix) {
        FloatArray[] weights = new FloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.floatBufferToFloatArray(tensorEntries.get(key));
        }
        return weights;
    }

    private HalfFloatArray[] loadLayerWeightsAsHalfFloatArrays(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, String layerName, String suffix) {
        HalfFloatArray[] weights = new HalfFloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadTensorAsHalfFloatArray(tensorEntries.get(key));
        }
        return weights;
    }
}
