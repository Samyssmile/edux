package de.example.knn;

import de.edux.ml.knn.ILabeledPoint;
import de.edux.ml.knn.KnnClassifier;
import de.edux.ml.knn.KnnPoint;
import de.example.data.seaborn.Penguin;
import de.example.data.seaborn.SeabornDataProcessor;
import de.example.data.seaborn.SeabornProvider;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Knn - K nearest neighbors
 * Dataset: Seaborn Penguins
 */
public class KnnSeabornExample {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;
    private static final boolean FILTER_INCOMPLETE_RECORDS = true;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.7;
    private static final File CSV_FILE = new File("example" + File.separator + "datasets" + File.separator + "seaborn-penguins" + File.separator + "penguins.csv");

    public static void main(String[] args) {
        //Load dataset
        var seabornDataProcessor = new SeabornDataProcessor();
        List<Penguin> dataset = seabornDataProcessor.loadTDataSet(CSV_FILE, ',', SHUFFLE, NORMALIZE, FILTER_INCOMPLETE_RECORDS);
        List<List<Penguin>> trainTestSplittedList = seabornDataProcessor.split(dataset, TRAIN_TEST_SPLIT_RATIO);
        var seabornProvider = new SeabornProvider(dataset, trainTestSplittedList.get(0), trainTestSplittedList.get(1));
        seabornProvider.printStatistics();

        // Train classifier
        List<ILabeledPoint> labeledPoints = new ArrayList<>();
        for (int i = 0; i < seabornProvider.getTrainFeatures().length; i++) {
            labeledPoints.add(new KnnPoint(seabornProvider.getTrainFeatures()[i], seabornProvider.getTrainData().get(i).species()));
        }

        KnnClassifier knnClassifier = new KnnClassifier(1, labeledPoints);

        // Evaluate classifier
        List<Penguin> testDataset = seabornProvider.getTestData();
        List<ILabeledPoint> testLabeledPoints = new ArrayList<>();
        testDataset.forEach(penguin -> {
            ILabeledPoint labeledPoint = new KnnPoint(penguin.getFeatures(), penguin.species());
            testLabeledPoints.add(labeledPoint);
        });

        knnClassifier.evaluate(testLabeledPoints);
    }

}
