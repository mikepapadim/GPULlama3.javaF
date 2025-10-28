package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.qwen3.Qwen3;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tokenizer.impl.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadQwen3Vocabulary;

public class Qwen3ModelLoader extends AbstractModelLoader<Qwen3, Qwen3Configuration> {

    public Qwen3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
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
    protected Qwen3Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen3.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;

        int vocabSize = metadata.containsKey("qwen3.vocab_size") ? (int) metadata.get("qwen3.vocab_size") : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new Qwen3Configuration((int) metadata.get("qwen3.embedding_length"), (int) metadata.get("qwen3.feed_forward_length"), (int) metadata.get("qwen3.block_count"),
                (int) metadata.get("qwen3.attention.head_count"),

                metadata.containsKey("qwen3.attention.head_count_kv") ? (int) metadata.get("qwen3.attention.head_count_kv") : (int) metadata.get("qwen3.attention.head_count"),
                (int) metadata.get("qwen3.attention.key_length"), (int) metadata.get("qwen3.attention.value_length"),

                vocabSize, modelContextLength, finalContextLength, false, (float) metadata.get("qwen3.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("qwen3.rope.freq_base"));
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen3Configuration config) {
        return RoPE.precomputeFreqsCis(config.contextLengthModel(), config.numberOfHeadsKey(), config.ropeTheta(), false, 0, 0, 0, 0);
    }

    @Override
    protected Qwen3 createModel(Qwen3Configuration config, Tokenizer tokenizer, Weights weights) {
        Map<String, Object> metadata = gguf.getMetadata();
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        // Qwen2.5-coder uses <|endoftext|> as stop-token.
        ChatTokens chatTokens = isDeepSeekR1DistillQwen ? new ChatTokens("<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "")
                : new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen3(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
        }
        return new Qwen3TornadoWeights(ModelLoader.loadTensorAsFloatArray(tokenEmbeddings), loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_norm", "weight"),
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_q", "weight"), loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_k", "weight"),
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_v", "weight"), loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "attn_output", "weight"),
                loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_k_norm", "weight"),   // attnKNorm
                loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "attn_q_norm", "weight"),   // attnQNorm
                loadLayerWeightsAsFloatArraysFromBuffer(tensorEntries, config, "ffn_norm", "weight"), loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_gate", "weight"),            // w1
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_down", "weight"),            // w2
                loadLayerWeightsAsHalfFloatArrays(tensorEntries, config, "ffn_up", "weight"),              // w3
                ModelLoader.floatBufferToFloatArray(tensorEntries.get("output_norm.weight")), FloatArray.fromArray(ropeFreqs.first()), FloatArray.fromArray(ropeFreqs.second()),
                ModelLoader.loadTensorAsHalfFloatArray(outputWeight), outputWeight.ggmlType());
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();
        return new Qwen3StandardWeights(ModelLoader.loadQuantized(tokenEmbeddings), loadLayerWeights(tensorEntries, config, "attn_norm", "weight"),    // rms_att_weight
                loadLayerWeights(tensorEntries, config, "attn_q", "weight"),       // wq
                loadLayerWeights(tensorEntries, config, "attn_k", "weight"),       // wk
                loadLayerWeights(tensorEntries, config, "attn_v", "weight"),       // wv
                loadLayerWeights(tensorEntries, config, "attn_output", "weight"),  // wo

                loadLayerWeights(tensorEntries, config, "attn_k_norm", "weight"),  // attnKNorm
                loadLayerWeights(tensorEntries, config, "attn_q_norm", "weight"),  // attnQNorm

                loadLayerWeights(tensorEntries, config, "ffn_norm", "weight"),     //rms_ffn_weight
                loadLayerWeights(tensorEntries, config, "ffn_gate", "weight"),     // w1
                loadLayerWeights(tensorEntries, config, "ffn_down", "weight"),     // w2
                loadLayerWeights(tensorEntries, config, "ffn_up", "weight"),       // w3
                ModelLoader.loadQuantized(tensorEntries.get("output_norm.weight")), // rms_final_weight
                new ArrayFloatTensor(ropeFreqsReal), new ArrayFloatTensor(ropeFreqsImag),
                tensorEntries.containsKey("output.weight") ? ModelLoader.loadQuantized(tensorEntries.get("output.weight")) : ModelLoader.loadQuantized(tokenEmbeddings), // weights are shared
                null);
    }

    // Helper methods
    private FloatTensor[] loadLayerWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, String layerName, String suffix) {
        FloatTensor[] weights = new FloatTensor[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadQuantized(tensorEntries.get(key));
        }
        return weights;
    }

    private FloatArray[] loadLayerWeightsAsFloatArraysFromBuffer(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, String layerName, String suffix) {
        FloatArray[] weights = new FloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.floatBufferToFloatArray(tensorEntries.get(key));
        }
        return weights;
    }

    private HalfFloatArray[] loadLayerWeightsAsHalfFloatArrays(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, String layerName, String suffix) {
        HalfFloatArray[] weights = new HalfFloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadTensorAsHalfFloatArray(tensorEntries.get(key));
        }
        return weights;
    }
}
