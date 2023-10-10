package de.example.knn;

import de.edux.api.Classifier;
import de.edux.ml.knn.KnnClassifier;
import de.edux.ml.nn.network.api.Dataset;
import de.example.data.seaborn.Penguin;
import de.example.data.seaborn.SeabornDataProcessor;
import de.example.data.seaborn.SeabornProvider;

import java.io.File;
import java.util.List;

/**
 * Knn - K nearest neighbors
 * Dataset: Seaborn Penguins
 */
public class KnnSeabornExample {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;
    private static final boolean FILTER_INCOMPLETE_RECORDS = true;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.75;
    private static final File CSV_FILE = new File("example" + File.separator + "datasets" + File.separator + "seaborn-penguins" + File.separator + "penguins.csv");

    public static void main(String[] args) {
        //Load Data, shuffle, normalize, filter incomplete records out.
        var seabornDataProcessor = new SeabornDataProcessor();
        List<Penguin> data = seabornDataProcessor.loadDataSetFromCSV(CSV_FILE, ',', SHUFFLE, NORMALIZE, FILTER_INCOMPLETE_RECORDS);
        //Split dataset into train and test
        Dataset<Penguin> dataset = seabornDataProcessor.split(data, TRAIN_TEST_SPLIT_RATIO);
        var seabornProvider = new SeabornProvider(data, dataset.trainData(), dataset.testData());
        seabornProvider.printStatistics();
        Classifier knn = new KnnClassifier(2);
        //Train and evaluate
        knn.train(seabornProvider.getTrainFeatures(), seabornProvider.getTrainLabels());
        knn.evaluate(seabornProvider.getTestFeatures(), seabornProvider.getTestLabels());
    }

}
