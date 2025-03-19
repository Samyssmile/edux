package de.example.cnn;

import de.edux.ml.api.ExecutionMode;
import de.edux.ml.mlp.core.network.NetworkBuilder;
import de.edux.ml.mlp.core.network.layers.*;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.fractality.FractalityLoader;

/**
 * @author Samuel Abramov
 */
public class CnnExampleOnFractality {
    static final int INPUT_HEIGHT = 64;
    static final int INPUT_WIDTH = 64;
    static final int BATCH_SIZE = 100; // Beibehaltung der Batch-Größe

    public static void main(String[] args) {
        new CnnExampleOnFractality().start();
    }

    private void start() {
        int numberOfOutputClasses = 6;
        int stride = 2;
        int filterSize = 3;
        int poolSize = 2;
        int epochs = 20; // Reduzierung der Epochenzahl
        float initialLearningRate = 0.01f;
        float finalLearningRate = 0.0001f;

        Loader trainLoader = new FractalityLoader(
                "C:\\Users\\windo\\Documents\\projects\\edux\\example\\datasets\\fractality-xs\\train",
                "C:\\Users\\windo\\Documents\\projects\\edux\\example\\datasets\\fractality-xs\\train\\images.csv",
                BATCH_SIZE,
                INPUT_HEIGHT,
                INPUT_WIDTH
        );

        Loader testLoader = new FractalityLoader(
                "C:\\Users\\windo\\Documents\\projects\\edux\\example\\datasets\\fractality-xs\\test",
                "C:\\Users\\windo\\Documents\\projects\\edux\\example\\datasets\\fractality-xs\\test\\images.csv",
                BATCH_SIZE,
                INPUT_HEIGHT,
                INPUT_WIDTH
        );


        long startTime = System.currentTimeMillis();

        new NetworkBuilder()
                .addLayer(new ConvolutionalLayer(16, filterSize, INPUT_HEIGHT, INPUT_WIDTH, 1)) // 8 Filter, 3x3, input 28x28, 1 grayscale channel
                .addLayer(new ReLuLayer())
                .addLayer(new PoolingLayer(16, 62, 62, poolSize, poolSize, stride)) // Pooling layer (2x2, stride 2)

                .addLayer(new FlattenLayer(16, 31, 31)) // Updated dimensions: 16 channels, 5x5 output
                .addLayer(new DenseLayer(16*31*31, numberOfOutputClasses)) // Final dense layer with number of classes as output
                .addLayer(new SoftmaxLayer())

                // Hyperparameter configuration
                .withBatchSize(BATCH_SIZE)
                .withLearningRates(initialLearningRate, finalLearningRate)
                .withExecutionMode(ExecutionMode.SINGLE_THREAD)
                .withEpochs(epochs)

                // Build network
                .build()
                .printArchitecture()
                .fit(trainLoader, testLoader)
                .saveModel("cnn_mnist_trained.edux");

        long endTime = System.currentTimeMillis();
        System.out.println("Das Training dauerte: " + (endTime - startTime) / 1000 + " Sekunden");
    }
}
