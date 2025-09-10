package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.inference.weights.Weights;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;

public abstract class TornadoWeightsQ8  implements Weights {
    public FloatArray[] rms_att_weightLayered;          // (layer, dim) rmsnorm weights
    public Int8Array[] wqLayered;                  // (layer, n_heads * head_size)
    public Int8Array[] wkLayered;                  // (layer, n_kv_heads, head_size)
    public Int8Array[] wvLayered;                  // (layer, n_kv_heads * head_size)
    public Int8Array[] woLayered;                  // (layer, n_heads * head_size, dim)
    public FloatArray[] rms_ffn_weightLayered;          // (layer, dim)
    public Int8Array[] w1Layered;                  // (layer, hidden_dim, dim)
    public Int8Array[] w2Layered;                  // (layer, dim, hidden_dim)
    public Int8Array[] w3Layered;                  // (layer, hidden_dim, dim)
    public FloatArray rms_final_weight_as_floatArray;
    public FloatArray tokenEmbeddingTable;              // (vocab_size, dim)
    public FloatArray freq_cis_realFlat;                // (seq_len, head_size/2)
    public FloatArray freq_cis_imagFlat;                // (seq_len, head_size/2)
    public Int8Array wclsHalfFloat;

    // (optional) classifier weights for the logits, on the last layer
    protected final GGMLType weightType;

    protected TornadoWeightsQ8(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            Int8Array[] wqLayered,
            Int8Array[] wkLayered,
            Int8Array[] wvLayered,
            Int8Array[] woLayered,
            FloatArray[] rms_ffn_weightLayered,
            Int8Array[] w1Layered,
            Int8Array[] w2Layered,
            Int8Array[] w3Layered,
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            Int8Array wclsByteArray,
            GGMLType weightType) {
        // TornadoVM format
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.rms_att_weightLayered = rms_att_weightLayered;
        this.wqLayered = wqLayered;
        this.wkLayered = wkLayered;
        this.wvLayered = wvLayered;
        this.woLayered = woLayered;
        this.rms_ffn_weightLayered = rms_ffn_weightLayered;
        this.w1Layered = w1Layered;
        this.w2Layered = w2Layered;
        this.w3Layered = w3Layered;
        this.rms_final_weight_as_floatArray = rms_final_weight_as_floatArray;
        this.freq_cis_realFlat = freq_cis_realFlat;
        this.freq_cis_imagFlat = freq_cis_imagFlat;
        this.wclsHalfFloat = wclsByteArray;
        this.weightType = weightType;
    }
    //@formatter:on

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }

}
