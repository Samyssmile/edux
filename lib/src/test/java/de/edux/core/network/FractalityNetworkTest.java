package de.edux.core.network;

import de.edux.ml.api.ExecutionMode;
import de.edux.ml.mlp.core.network.NetworkBuilder;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.core.network.loader.fractality.FractalityLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class FractalityNetworkTest {

  private static FractalityLoader fractalityTrainLoader;
  private static FractalityLoader fractalityTestLoader;

  @BeforeAll
  static void setUp() {
    fractalityTrainLoader =
        new FractalityLoader(
            "src/test/resources/fractality/train/class",
            "src/test/resources/fractality/train/images.csv",
            5,
            256,
            256);

    fractalityTestLoader =
        new FractalityLoader(
            "src/test/resources/fractality/test/class",
            "src/test/resources/fractality/test/images.csv",
            5,
            256,
            256);
  }

  @Test
  public void shouldTrain() {
    int batchSize = 5;
    ExecutionMode singleThread = ExecutionMode.SINGLE_THREAD;
    int epochs = 5;
    float initialLearningRate = 0.1f;
    float finalLearningRate = 0.001f;

    MetaData trainMetaData = fractalityTrainLoader.open();
    int inputSize = trainMetaData.getInputSize();
    int outputSize = trainMetaData.getNumberOfClasses();
    fractalityTrainLoader.close();

    // Training from scratch
    new NetworkBuilder()
        .addLayer(new DenseLayer(inputSize, 256))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(256, 256))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(256, 256))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(256, outputSize))
        .addLayer(new SoftmaxLayer())
        .withBatchSize(batchSize)
        .withLearningRates(initialLearningRate, finalLearningRate)
        .withExecutionMode(singleThread)
        .withEpochs(epochs)
        .build()
        .printArchitecture()
        .fit(fractalityTrainLoader, fractalityTestLoader)
        .saveModel("fractality.edux");
  }
}
