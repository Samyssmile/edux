package de.edux.ml;

import de.edux.api.Classifier;
import de.edux.data.handler.EIncompleteRecordsHandlerStrategy;
import de.edux.data.provider.SeabornDataProcessor;
import de.edux.data.provider.SeabornProvider;
import de.edux.ml.randomforest.RandomForest;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URL;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RandomForestTest {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;
    private static final EIncompleteRecordsHandlerStrategy INCOMPLETE_RECORD_HANDLER_STRATEGY = EIncompleteRecordsHandlerStrategy.DROP_RECORDS;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.7;
    private static final String CSV_FILE_PATH = "testdatasets/seaborn-penguins/penguins.csv";
    private static SeabornProvider seabornProvider;
    @BeforeAll
    static void setup() {
        URL url = RandomForestTest.class.getClassLoader().getResource(CSV_FILE_PATH);
        if (url == null) {
            throw new IllegalStateException("Cannot find file: " + CSV_FILE_PATH);
        }
        File csvFile = new File(url.getPath());
        var seabornDataProcessor = new SeabornDataProcessor();
        var dataset = seabornDataProcessor.loadDataSetFromCSV(csvFile, ',', SHUFFLE, NORMALIZE, INCOMPLETE_RECORD_HANDLER_STRATEGY);
        var splitedDataset = seabornDataProcessor.split(dataset, TRAIN_TEST_SPLIT_RATIO);
        seabornProvider = new SeabornProvider(dataset, splitedDataset.trainData(), splitedDataset.testData());
    }
    @Test
    void train() {
        double[][] features = seabornProvider.getTrainFeatures();
        double[][] labels = seabornProvider.getTrainLabels();

        double[][] testFeatures = seabornProvider.getTestFeatures();
        double[][] testLabels = seabornProvider.getTestLabels();

        assertTrue(features.length > 0);
        assertTrue(labels.length > 0);
        assertTrue(testFeatures.length > 0);
        assertTrue(testLabels.length > 0);

        int numberOfTrees = 100;
        int maxDepth = 8;
        int minSampleSize = 2;
        int minSamplesLeaf = 1;
        int maxLeafNodes = 2;
        int numFeatures = (int) Math.sqrt(features.length)*3;

        Classifier randomForest = new RandomForest( numberOfTrees, maxDepth, minSampleSize, minSamplesLeaf, maxLeafNodes,numFeatures);
        randomForest.train( features, labels);
        double accuracy = randomForest.evaluate(testFeatures, testLabels);
        System.out.println(accuracy);
        assertTrue(accuracy>0.7);
    }
}