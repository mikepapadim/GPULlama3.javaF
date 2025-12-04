package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class Q8_0TornadoTensor extends TornadoTensor {

    private final int size;
    private final HalfFloatArray scales;  // One per 32-element block
    private final Int8Array quants;       // Quantized int8 values
    private MemorySegment segment;

    private final ByteArray tornadoNativeArray;

    public Q8_0TornadoTensor(int size, HalfFloatArray scales, Int8Array quants, MemorySegment segment) {
        this.size = size;
        this.scales = scales;
        this.quants = quants;
        this.segment = segment;
        this.tornadoNativeArray = null;
    }

    public Q8_0TornadoTensor(ByteArray byteArray) {
        this.size = -1;
        this.scales = null;
        this.quants = null;
        this.segment = null;
        this.tornadoNativeArray = byteArray;
    }

    public static Q8_0TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new Q8_0TornadoTensor(ByteArray.fromSegmentShallow(segment));
    }

    public int getSize() {
        return size;
    }

    /**
     * Returns the scale factors for GPU kernels.
     *
     * @return HalfFloatArray containing fp16 scale factors
     */
    public HalfFloatArray getScales() {
        return scales;
    }

    /**
     * Returns the quantized values for GPU kernels.
     *
     * @return Int8Array containing quantized int8 values
     */
    public Int8Array getQuants() {
        return quants;
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    public MemorySegment asMemorySegment() {
        return segment;
    }

    /**
     * Dequantizes and returns a single float value.
     *
     * @param index Element index
     * @return Dequantized float value
     */
    public float getFloat(int index) {
        assert 0 <= index;
        int blockIdx = index / GGMLType.Q8_0.getBlockSize();
        float scale = scales.get(blockIdx).getFloat32();
        byte quant = quants.get(index);
        return quant * scale;
    }

}
