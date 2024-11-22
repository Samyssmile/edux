package de.edux.core.network;

import java.nio.file.Paths;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import de.edux.ml.api.ExecutionMode;
import de.edux.ml.mlp.core.network.NetworkBuilder;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.core.network.loader.fractality.FractalityLoader;

public class FractalityNetworkTest {

  private static Loader fractalityTrainLoader;
  private static Loader fractalityTestLoader;

  @BeforeAll
  static void setUp() {

    fractalityTrainLoader =
        new FractalityLoader(
            Paths.get("src", "test", "resources", "fractality", "train", "class").toString(),
            Paths.get("src", "test", "resources", "fractality", "train", "images.csv").toString(),
            100,
            256,
                256);

    fractalityTestLoader = new FractalityLoader(
        Paths.get("src", "test", "resources", "fractality", "test", "class").toString(),
        Paths.get("src", "test", "resources", "fractality", "test", "images.csv").toString(),
        10,
            256,
            256);
}

  @Test
  public void shouldTrain() {
    int batchSize = 100;
    ExecutionMode singleThread = ExecutionMode.SINGLE_THREAD;
    int epochs = 100;
    float initialLearningRate = 0.01f;
    float finalLearningRate = 0.0001f;

    MetaData trainMetaData = fractalityTrainLoader.open();
    int inputSize = trainMetaData.getInputSize();
    int outputSize = trainMetaData.getNumberOfClasses();
    fractalityTrainLoader.close();

    // Training from scratch
    new NetworkBuilder()
        .addLayer(new DenseLayer(inputSize, 32))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(32, outputSize))
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
