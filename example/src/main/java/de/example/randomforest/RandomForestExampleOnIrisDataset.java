package de.example.randomforest;

import de.edux.api.Classifier;
import de.edux.data.provider.DataProcessor;
import de.edux.data.reader.CSVIDataReader;
import de.edux.ml.randomforest.RandomForest;
import java.io.File;

public class RandomForestExampleOnIrisDataset {

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
    /*   IRIS Dataset...
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
    /* train 100 decision trees with max depth of 64, min samples split of 2, min samples leaf of 1, max features of 3 and 50 of samples */
    Classifier randomForest = new RandomForest(1000, 64, 2, 2, 3, 50);

    var trainFeatures = irisDataProcessor.getTrainFeatures(featureColumnIndices);
    var trainTestFeatures = irisDataProcessor.getTestFeatures(featureColumnIndices);
    var trainLabels = irisDataProcessor.getTrainLabels(targetColumnIndex);
    var trainTestLabels = irisDataProcessor.getTestLabels(targetColumnIndex);

    randomForest.train(trainFeatures, trainLabels);
    randomForest.evaluate(trainTestFeatures, trainTestLabels);
  }
}
