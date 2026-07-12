package org.beehive.gpullama3.bench;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/** Prints Gemma4 config dims + PLE weight precisions (to size batch buffers + pick GEMM paths). */
public class GemmaProbe {
    public static void main(String[] args) throws Exception {
        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        Gemma4Configuration c = (Gemma4Configuration) model.configuration();
        Gemma4TornadoWeights w = (Gemma4TornadoWeights) model.weights();
        System.out.printf("dim=%d layers=%d nHeads=%d nHeadKv=%d headDimSwa=%d headDimFull=%d%n",
                c.dim(), c.numberOfLayers(), c.numberOfHeads(), c.numberOfKeyValueHeads(), c.headDimSwa(), c.headDimFull());
        System.out.printf("nEmbdPerLayer=%d sharedKvLayers=%d slidingWindow=%d vocab=%d%n",
                c.embeddingLengthPerLayer(), c.sharedKvLayers(), c.slidingWindowSize(), c.vocabularySize());
        System.out.print("ffnLen[0..]=");
        for (int i = 0; i < Math.min(6, c.numberOfLayers()); i++) System.out.print(c.feedForwardLength(i) + " ");
        System.out.println();
        System.out.print("swaPattern[0..]=");
        for (int i = 0; i < Math.min(8, c.numberOfLayers()); i++) System.out.print(c.isSwa(i) + " ");
        System.out.println();
        System.out.printf("perLayerInpGate[0]=%s perLayerProj[0]=%s perLayerModelProj=%s wq[0]=%s wcls=%s%n",
                w.perLayerInpGate[0].type(), w.perLayerProj[0].type(), w.perLayerModelProj.type(),
                w.wqLayered[0].type(), w.wclsByteArray.type());
    }
}
