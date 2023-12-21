package de.example.mlp;

import de.edux.ml.api.ExecutionMode;
import de.edux.ml.mlp.core.network.NetworkBuilder;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.core.network.loader.mnist.MnistLoader;
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
    int epochs = 100;
    float initialLearningRate = 0.1f;
    float finalLearningRate = 0.0001f;

    Loader trainLoader = new MnistLoader(trainImages, trainLabels, batchSize);
    Loader testLoader = new MnistLoader(testImages, testLabels, batchSize);

    MetaData trainMetaData = trainLoader.open();
    int inputSize = trainMetaData.getInputSize();
    int outputSize = trainMetaData.getNumberOfClasses();
    trainLoader.close();

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
        .fit(trainLoader, testLoader)
        .saveModel("mnist_trained.edux");

    // Loading a trained model
    new NetworkBuilder()
        .withExecutionMode(singleThread)
        .withEpochs(5)
        .withLearningRates(0.001f, 0.001f)
        .loadModel("mnist_trained.edux")
        .fit(trainLoader, testLoader);
  }
}
