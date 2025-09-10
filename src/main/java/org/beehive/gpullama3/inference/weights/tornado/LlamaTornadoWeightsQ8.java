package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;

public class LlamaTornadoWeightsQ8 extends TornadoWeightsQ8 {
    // @formatter:off
    public LlamaTornadoWeightsQ8(
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
        // call to TornadoWeights constructor
        super(tokenEmbeddingTable,
                rms_att_weightLayered,
                wqLayered,
                wkLayered,
                wvLayered,
                woLayered,
                rms_ffn_weightLayered,
                w1Layered,
                w2Layered,
                w3Layered,
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);
    }
    // @formatter:on
}
