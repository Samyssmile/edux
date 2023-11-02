package de.example.nn;

import de.edux.data.provider.DataProcessor;
import de.edux.data.reader.CSVIDataReader;
import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.imputation.ImputationStrategy;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import java.io.File;
import java.util.List;

public class MultilayerNeuralNetworkExampleOnPenguinsDataset {

  private static final double TRAIN_TEST_SPLIT_RATIO = 0.70;
  private static final File CSV_FILE =
      new File(
          "example"
              + File.separator
              + "datasets"
              + File.separator
              + "seaborn-penguins"
              + File.separator
              + "penguins.csv");
  private static final boolean SKIP_HEAD = true;
  private static final ImputationStrategy averageImputation = ImputationStrategy.AVERAGE;
  private static final ImputationStrategy modeImputation = ImputationStrategy.MODE;

  public static void main(String[] args) {
    /*   Penguins Dataset...
        +--------+--------+---------------+--------------+------------------+------------------+
        | species|  island| bill_length_mm| bill_depth_mm| flipper_length_mm| body_mass_g|  sex|
        +--------+--------+---------------+--------------+------------------+------------------+
        | Gentoo | Biscoe |      49.6     |      16      |         225      |    5700    | MALE|
        +--------+--------+---------------+--------------+------------------+------------------+
    */

    var featureColumnIndices = new int[] {1, 2, 3, 4, 5, 6};
    var targetColumnIndex = 0;

    var penguinsDataProcessor =
        new DataProcessor(new CSVIDataReader())
            .loadDataSetFromCSV(CSV_FILE, ',', SKIP_HEAD, featureColumnIndices, targetColumnIndex)
            .imputation(0, modeImputation)
            .imputation(1, modeImputation)
            .imputation(2, averageImputation)
            .imputation(3, averageImputation)
            .imputation(4, averageImputation)
            .imputation(5, averageImputation)
            .imputation(6, modeImputation)
            .normalize()
            .shuffle()
            .split(TRAIN_TEST_SPLIT_RATIO);

    var trainFeatures = penguinsDataProcessor.getTrainFeatures(featureColumnIndices);
    var trainLabels = penguinsDataProcessor.getTrainLabels(targetColumnIndex);
    var testFeatures = penguinsDataProcessor.getTestFeatures(featureColumnIndices);
    var testLabels = penguinsDataProcessor.getTestLabels(targetColumnIndex);

    var classMap = penguinsDataProcessor.getClassMap();

    System.out.println("Class Map: " + classMap);

    // Configure Network with:
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
