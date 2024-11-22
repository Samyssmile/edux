package de.example.cnn;

import de.edux.ml.api.ExecutionMode;
import de.edux.ml.mlp.core.network.NetworkBuilder;
import de.edux.ml.mlp.core.network.layers.ConvolutionalLayer;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.FlattenLayer;
import de.edux.ml.mlp.core.network.layers.PoolingLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.core.network.loader.mnist.MnistLoader;

import java.io.File;

/**
 * @author Samuel Abramov
 */
public class CnnExampleOnMNIST {
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
        int epochs = 2;
        float initialLearningRate = 0.1f;
        float finalLearningRate = 0.0001f;

        Loader trainLoader = new MnistLoader(trainImages, trainLabels, batchSize);
        Loader testLoader = new MnistLoader(testImages, testLabels, batchSize);
        MetaData trainMetaData = trainLoader.open();
        int inputSize = trainMetaData.getInputSize();
        int numberOfOutputClasses = trainMetaData.getNumberOfClasses();
        trainLoader.close();

        long startTime = System.currentTimeMillis();
        new NetworkBuilder()
                .addLayer(new ConvolutionalLayer(8, 3, 28, 28, 1)) // 8 Filter, 3x3, input 28x28, 1 grayscale channel
                .addLayer(new ReLuLayer())
                .addLayer(new PoolingLayer(8, 26, 26, 2, 2, 2)) // Pooling layer (2x2, stride 2)

                .addLayer(new FlattenLayer(8, 13, 13)) // Updated dimensions: 16 channels, 5x5 output
                .addLayer(new DenseLayer(8*13*13, numberOfOutputClasses)) // Final dense layer with number of classes as output
                .addLayer(new SoftmaxLayer())

                // Hyperparameter configuration
                .withBatchSize(batchSize)
                .withLearningRates(initialLearningRate, finalLearningRate)
                .withExecutionMode(ExecutionMode.SINGLE_THREAD)
                .withEpochs(epochs)

                // Build network
                .build()
                .printArchitecture()
                .fit(trainLoader, testLoader)
                .saveModel("cnn_mnist_trained.edux");

        long endTime = System.currentTimeMillis();
        System.out.println("Training took: " + (endTime - startTime) / 1000 + " seconds");

    }
}
