package de.example.nn;

import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import de.example.data.IrisProvider;

import java.util.List;

import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import de.example.data.IrisProvider;

import java.util.List;

public class MultilayerPerceptronExample {

    private static final boolean SHUFFLE = true;
    private final static boolean NORMALIZE = true;

    public static void main(String[] args) {
        // Get IRIS dataset
        var datasetProvider = new IrisProvider(NORMALIZE, SHUFFLE, 0.7);
        datasetProvider.printStatistics();

        //Get Features and Labels
        double[][] features = datasetProvider.getTrainFeatures();
        double[][] labels = datasetProvider.getTrainLabels();

        //Get Test Features and Labels
        double[][] testFeatures = datasetProvider.getTestFeatures();
        double[][] testLabels = datasetProvider.getTestLabels();

        //Configure Network with:
        // - 4 Input Neurons
        // - 2 Hidden Layer with 12 and 6 Neurons
        // - 3 Output Neurons
        // - Learning Rate of 0.1
        NetworkConfiguration networkConfiguration = new NetworkConfiguration(features[0].length, List.of(12, 6), 3, 0.01, 1000, ActivationFunction.LEAKY_RELU, ActivationFunction.SOFTMAX, LossFunction.CATEGORICAL_CROSS_ENTROPY);

        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron(features, labels, testFeatures, testLabels, networkConfiguration);
        multilayerPerceptron.train();
        multilayerPerceptron.evaluate(testFeatures, testLabels);
    }
}

