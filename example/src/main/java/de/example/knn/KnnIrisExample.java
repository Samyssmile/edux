package de.example.knn;

import de.edux.api.Classifier;
import de.edux.data.provider.DataProcessor;
import de.edux.data.reader.CSVIDataReader;
import de.edux.ml.knn.KnnClassifier;

import java.io.File;

public class KnnIrisExample {
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.70;
    private static final File CSV_FILE = new File("example" + File.separator + "datasets" + File.separator + "iris" + File.separator + "iris.csv");
    private static final boolean SKIP_HEAD = true;

    public static void main(String[] args) {
        var featureColumnIndices = new int[]{0, 1, 2, 3};
        var targetColumnIndex = 4;

        var irisDataProcessor = new DataProcessor(new CSVIDataReader())
                .loadDataSetFromCSV(CSV_FILE, ',', SKIP_HEAD, featureColumnIndices, targetColumnIndex)
                .normalize()
                .shuffle()
                .split(TRAIN_TEST_SPLIT_RATIO);


        Classifier knn = new KnnClassifier(2);

        var trainFeatures = irisDataProcessor.getTrainFeatures(featureColumnIndices);
        var trainTestFeatures = irisDataProcessor.getTestFeatures(featureColumnIndices);
        var trainLabels = irisDataProcessor.getTrainLabels(targetColumnIndex);
        var trainTestLabels = irisDataProcessor.getTestLabels(targetColumnIndex);

        knn.train(trainFeatures, trainLabels);
        knn.evaluate(trainTestFeatures, trainTestLabels);
    }
}
