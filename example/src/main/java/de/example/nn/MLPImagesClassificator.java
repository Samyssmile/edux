package de.example.nn;

import de.edux.data.provider.experimental.ImageDatasetProvider;
import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import java.util.List;

public class MLPImagesClassificator {

  public static void main(String[] args) {
    ImageDatasetProvider imagedatasetProvider = new ImageDatasetProvider();
    double[][] features = imagedatasetProvider.getTrainFeatures();
    double[][] labels = imagedatasetProvider.getTrainLabels();

    double[][] testFeatures = imagedatasetProvider.getTestFeatures();
    double[][] testLabels = imagedatasetProvider.getTestLabels();

    System.out.println("Number of training images: " + features.length);
    System.out.println("Number of test images: " + testFeatures.length);

    var networkConfiguration =
        new NetworkConfiguration(
            features[0].length,
            List.of(features[0].length, features[0].length * 2, features[0].length * 4),
            labels[0].length,
            0.01,
            300,
            ActivationFunction.LEAKY_RELU,
            ActivationFunction.SOFTMAX,
            LossFunction.CATEGORICAL_CROSS_ENTROPY,
            Initialization.XAVIER,
            Initialization.XAVIER);

    System.out.println("Network configuration: " + networkConfiguration);
    MultilayerPerceptron multilayerPerceptron =
        new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);
    System.out.println("Training started...");
    multilayerPerceptron.train(features, labels);
  }
}
