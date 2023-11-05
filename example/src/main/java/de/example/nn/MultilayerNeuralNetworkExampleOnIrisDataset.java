package de.example.nn;

import de.edux.data.provider.DataProcessor;
import de.edux.data.reader.CSVIDataReader;
import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import java.io.File;
import java.util.List;

public class MultilayerNeuralNetworkExampleOnIrisDataset {

  private static final double TRAIN_TEST_SPLIT_RATIO = 0.70;
  private static final File CSV_FILE =
      new File(
          "example"
              + File.separator
              + "datasets"
              + File.separator
              + "iris"
              + File.separator
              + "iris.csv");
  private static final boolean SKIP_HEAD = true;

  public static void main(String[] args) {
    var featureColumnIndices = new int[] {0, 1, 2, 3};
    var targetColumnIndex = 4;

    var dataProcessor = new DataProcessor(new CSVIDataReader());
    var dataset =
        dataProcessor.loadDataSetFromCSV(
            CSV_FILE, ',', SKIP_HEAD, featureColumnIndices, targetColumnIndex);
    dataset.shuffle();
    dataset.normalize();
    dataProcessor.split(TRAIN_TEST_SPLIT_RATIO);

    var trainFeatures = dataProcessor.getTrainFeatures(featureColumnIndices);
    var trainLabels = dataProcessor.getTrainLabels(targetColumnIndex);
    var testFeatures = dataProcessor.getTestFeatures(featureColumnIndices);
    var testLabels = dataProcessor.getTestLabels(targetColumnIndex);

    var classMap = dataProcessor.getClassMap();

    System.out.println("Class Map: " + classMap);

    // Configure Network with:
    // - 4 Input Neurons
    // - 2 Hidden Layer with 12 and 6 Neurons
    // - 3 Output Neurons
    // - Learning Rate of 0.1
    // - 300 Epochs
    // - Leaky ReLU as Activation Function for Hidden Layers
    // - Softmax as Activation Function for Output Layer
    // - Categorical Cross Entropy as Loss Function
    // - Xavier as Weight Initialization for Hidden Layers
    // - Xavier as Weight Initialization for Output Layer
    var networkConfiguration =
        new NetworkConfiguration(
            trainFeatures[0].length,
            List.of(128, 256, 512),
            3,
            0.01,
            300,
            ActivationFunction.LEAKY_RELU,
            ActivationFunction.SOFTMAX,
            LossFunction.CATEGORICAL_CROSS_ENTROPY,
            Initialization.XAVIER,
            Initialization.XAVIER);

    MultilayerPerceptron multilayerPerceptron =
        new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);
    multilayerPerceptron.train(trainFeatures, trainLabels);
    multilayerPerceptron.evaluate(testFeatures, testLabels);
  }
}
