package de.example.knn;

import de.edux.api.Classifier;
import de.edux.data.handler.EIncompleteRecordsHandlerStrategy;
import de.edux.ml.knn.KnnClassifier;
import de.example.data.iris.Iris;
import de.example.data.iris.IrisDataProcessor;

import java.io.File;
import java.util.List;

/**
 * Knn - K nearest neighbors
 * Dataset: Iris
 * First transfer the iris data into KnnPoints, use the variety as label. Then use the KnnClassifier to classify the test data.
 */
public class KnnIrisExample {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;
    private static final EIncompleteRecordsHandlerStrategy INCOMPLETE_RECORD_HANDLER_STRATEGY = EIncompleteRecordsHandlerStrategy.DROP_RECORDS;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.75;
    private static final File CSV_FILE = new File("example" + File.separator + "datasets" + File.separator + "iris" + File.separator + "iris.csv");

    public static void main(String[] args) {
        var irisDataProcessor = new IrisDataProcessor();
        List<Iris> data = irisDataProcessor.loadDataSetFromCSV(CSV_FILE, ',', SHUFFLE, NORMALIZE, INCOMPLETE_RECORD_HANDLER_STRATEGY);
        irisDataProcessor.split(data, TRAIN_TEST_SPLIT_RATIO);

        Classifier knn = new KnnClassifier(2);

        knn.train(irisDataProcessor.getTrainFeatures(), irisDataProcessor.getTrainLabels());
        knn.evaluate(irisDataProcessor.getTestFeatures(), irisDataProcessor.getTestLabels());
    }
}