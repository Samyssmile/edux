package de.example.nn;

import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import de.example.data.iris.IrisProvider;

import java.util.List;

public class MultilayerPerceptronExample {

    private static final boolean SHUFFLE = true;
    private final static boolean NORMALIZE = true;

    public static void main(String[] args) {
        var datasetProvider = new IrisProvider(NORMALIZE, SHUFFLE, 0.7);
        datasetProvider.printStatistics();

        double[][] features = datasetProvider.getTrainFeatures();
        double[][] labels = datasetProvider.getTrainLabels();

        double[][] testFeatures = datasetProvider.getTestFeatures();
        double[][] testLabels = datasetProvider.getTestLabels();

        //Configure Network with:
        // - 4 Input Neurons
        // - 2 Hidden Layer with 12 and 6 Neurons
        // - 3 Output Neurons
        // - Learning Rate of 0.1
        // - 1000 Epochs
        // - Leaky ReLU as Activation Function for Hidden Layers
        // - Softmax as Activation Function for Output Layer
        // - Categorical Cross Entropy as Loss Function
        // - Xavier as Weight Initialization for Hidden Layers
        // - Xavier as Weight Initialization for Output Layer
        NetworkConfiguration networkConfiguration = new NetworkConfiguration(features[0].length, List.of(32, 6), 3, 0.01, 1000, ActivationFunction.LEAKY_RELU, ActivationFunction.SOFTMAX, LossFunction.CATEGORICAL_CROSS_ENTROPY, Initialization.XAVIER, Initialization.XAVIER);

        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);
        multilayerPerceptron.train(features, labels);
        multilayerPerceptron.evaluate(testFeatures, testLabels);
    }
}

