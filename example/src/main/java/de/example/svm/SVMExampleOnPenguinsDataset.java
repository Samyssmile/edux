package de.example.svm;

import de.edux.api.Classifier;
import de.edux.data.provider.DataProcessor;
import de.edux.data.reader.CSVIDataReader;
import de.edux.functions.imputation.ImputationStrategy;
import de.edux.ml.svm.SVMKernel;
import de.edux.ml.svm.SupportVectorMachine;
import java.io.File;

public class SVMExampleOnPenguinsDataset {
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

    Classifier svm = new SupportVectorMachine(SVMKernel.LINEAR, 2);

    var trainFeatures = penguinsDataProcessor.getTrainFeatures(featureColumnIndices);
    var trainTestFeatures = penguinsDataProcessor.getTestFeatures(featureColumnIndices);
    var trainLabels = penguinsDataProcessor.getTrainLabels(targetColumnIndex);
    var trainTestLabels = penguinsDataProcessor.getTestLabels(targetColumnIndex);

    svm.train(trainFeatures, trainLabels);
    svm.evaluate(trainTestFeatures, trainTestLabels);
  }
}
