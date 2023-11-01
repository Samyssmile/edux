package de.example.svm;

import de.edux.api.Classifier;
import de.edux.data.provider.DataProcessor;
import de.edux.data.reader.CSVIDataReader;
import de.edux.ml.svm.SVMKernel;
import de.edux.ml.svm.SupportVectorMachine;
import java.io.File;

public class SVMExampleOnIrisDataset {
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
    /* IRIS Dataset...*/
    /*
        +-------------+------------+-------------+------------+---------+
        | sepal.length| sepal.width| petal.length| petal.width| variety |
        +-------------+------------+-------------+------------+---------+
        |     5.1     |     3.5    |     1.4     |     .2     | Setosa  |
        +-------------+------------+-------------+------------+---------+
    */

    var featureColumnIndices = new int[] {0, 1, 2, 3}; // First 4 columns are features
    var targetColumnIndex = 4; // Last column is the target

    var irisDataProcessor =
        new DataProcessor(new CSVIDataReader())
            .loadDataSetFromCSV(CSV_FILE, ',', SKIP_HEAD, featureColumnIndices, targetColumnIndex)
            .normalize()
            .shuffle()
            .split(TRAIN_TEST_SPLIT_RATIO);

    Classifier svm = new SupportVectorMachine(SVMKernel.LINEAR, 2);

    var trainFeatures = irisDataProcessor.getTrainFeatures(featureColumnIndices);
    var trainTestFeatures = irisDataProcessor.getTestFeatures(featureColumnIndices);
    var trainLabels = irisDataProcessor.getTrainLabels(targetColumnIndex);
    var trainTestLabels = irisDataProcessor.getTestLabels(targetColumnIndex);

    svm.train(trainFeatures, trainLabels);
    svm.evaluate(trainTestFeatures, trainTestLabels);
  }
}
