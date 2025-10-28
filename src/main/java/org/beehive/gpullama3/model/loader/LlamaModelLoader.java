package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.LlamaStandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.llama.Llama;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tokenizer.impl.LlamaTokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

public class LlamaModelLoader extends AbstractModelLoader<Llama, LlamaConfiguration> {

    public LlamaModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadLlamaVocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new LlamaTokenizer(metadata, vocabulary);
    }

    @Override
    protected LlamaConfiguration createConfiguration(Map<String, Object> metadata) {
        int vocabSize = metadata.containsKey("llama.vocab_size") ? (int) metadata.get("llama.vocab_size") : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new LlamaConfiguration((int) metadata.get("llama.embedding_length"), (int) metadata.get("llama.feed_forward_length"), (int) metadata.get("llama.block_count"),
                (int) metadata.get("llama.attention.head_count"),
                metadata.containsKey("llama.attention.head_count_kv") ? (int) metadata.get("llama.attention.head_count_kv") : (int) metadata.get("llama.attention.head_count"), vocabSize,
                (int) metadata.get("llama.context_length"), (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)).withContextLength(contextLength);
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(LlamaConfiguration config) {
        return RoPE.precomputeFreqsCis(config.contextLength(), config.dim() / config.numberOfHeads(), config.ropeTheta(), false, 1.0f, 1.0f, 1.0f, config.contextLength()
        );
    }

    @Override
    protected Llama createModel(LlamaConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Llama(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, LlamaConfiguration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        return new LlamaStandardWeights(ModelLoader.loadQuantized(tokenEmbeddings),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                ModelLoader.loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                ModelLoader.loadQuantized(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                ModelLoader.loadQuantized(outputWeight),
                outputWeight.ggmlType());
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, LlamaConfiguration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        return new LlamaTornadoWeights(ModelLoader.loadTensorAsFloatArray(tokenEmbeddings),
                ModelLoader.loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                ModelLoader.loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                ModelLoader.loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                ModelLoader.loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                ModelLoader.loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                ModelLoader.loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                ModelLoader.loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                ModelLoader.loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                ModelLoader.loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                ModelLoader.floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()),
                FloatArray.fromArray(ropeFreqs.second()),
                ModelLoader.loadTensorAsHalfFloatArray(outputWeight),
                outputWeight.ggmlType());
    }
}
