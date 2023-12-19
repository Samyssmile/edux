package de.example.mlp;

import de.edux.ml.api.ExecutionMode;
import de.edux.ml.mlp.core.network.NetworkBuilder;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.image.ImageLoader;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import java.io.File;

public class MlpExampleOnMNIST {
  public static void main(String[] args) {
    String trainImages =
        "example"
            + File.separator
            + "datasets"
            + File.separator
            + "mnist"
            + File.separator
            + "train-images.idx3-ubyte";
    String trainLabels =
        "example"
            + File.separator
            + "datasets"
            + File.separator
            + "mnist"
            + File.separator
            + "train-labels.idx1-ubyte";
    String testImages =
        "example"
            + File.separator
            + "datasets"
            + File.separator
            + "mnist"
            + File.separator
            + "t10k-images.idx3-ubyte";
    String testLabels =
        "example"
            + File.separator
            + "datasets"
            + File.separator
            + "mnist"
            + File.separator
            + "t10k-labels.idx1-ubyte";

    int batchSize = 100;
    ExecutionMode singleThread = ExecutionMode.SINGLE_THREAD;
    int epochs = 5;
    float initialLearningRate = 0.1f;
    float finalLearningRate = 0.001f;

    Loader trainLoader = new ImageLoader(trainImages, trainLabels, batchSize);
    Loader testLoader = new ImageLoader(testImages, testLabels, batchSize);

    MetaData trainMetaData = trainLoader.open();
    int inputSize = trainMetaData.getInputSize();
    int outputSize = trainMetaData.getExpectedSize();
    trainLoader.close();

    // Training from scratch
    new NetworkBuilder()
        .addLayer(new DenseLayer(inputSize, 128))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(128, 128))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(128, outputSize))
        .addLayer(new SoftmaxLayer())
        .withBatchSize(batchSize)
        .withLearningRates(initialLearningRate, finalLearningRate)
        .withExecutionMode(singleThread)
        .withEpochs(epochs)
        .build()
        .printArchitecture()
        .fit(trainLoader, testLoader)
        .saveModel("mnist_trained.edux");

    // Loading a trained model
    new NetworkBuilder()
        .withExecutionMode(singleThread)
        .withEpochs(5)
        .withLearningRates(0.01f, 0.001f)
        .loadModel("mnist_trained.edux")
        .fit(trainLoader, testLoader);
  }
}
