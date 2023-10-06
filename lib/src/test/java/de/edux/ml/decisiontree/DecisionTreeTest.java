package de.edux.ml.decisiontree;

import de.edux.data.provider.Penguin;
import de.edux.data.provider.SeabornDataProcessor;
import de.edux.data.provider.SeabornProvider;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URL;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DecisionTreeTest {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;
    private static final boolean FILTER_INCOMPLETE_RECORDS = true;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.7;
    private static final String CSV_FILE_PATH = "testdatasets/seaborn-penguins/penguins.csv";
    private static SeabornProvider seabornProvider;
    @BeforeAll
    static void setup() {
        URL url = DecisionTreeTest.class.getClassLoader().getResource(CSV_FILE_PATH);
        if (url == null) {
            throw new IllegalStateException("Cannot find file: " + CSV_FILE_PATH);
        }
        File csvFile = new File(url.getPath());
        var seabornDataProcessor = new SeabornDataProcessor();
        var dataset = seabornDataProcessor.loadTDataSet(csvFile, ',', SHUFFLE, NORMALIZE, FILTER_INCOMPLETE_RECORDS);
        List<List<Penguin>> trainTestSplittedList = seabornDataProcessor.split(dataset, TRAIN_TEST_SPLIT_RATIO);
        seabornProvider = new SeabornProvider(dataset, trainTestSplittedList.get(0), trainTestSplittedList.get(1));
    }

    @RepeatedTest(5)
    void train() {
        double[][] features = seabornProvider.getTrainFeatures();
        double[][] labels = seabornProvider.getTrainLabels();

        double[][] testFeatures = seabornProvider.getTestFeatures();
        double[][] testLabels = seabornProvider.getTestLabels();

        assertTrue(features.length > 0);
        assertTrue(labels.length > 0);
        assertTrue(testFeatures.length > 0);
        assertTrue(testLabels.length > 0);

        IDecisionTree decisionTree = new DecisionTree();
        decisionTree.train(features, labels, 10, 2, 1, 8);
        double accuracy = decisionTree.evaluate(testFeatures, testLabels);
        assertTrue(accuracy>0.7);
    }

    @Test
    void predict() {
    }

    @Test
    void evaluate() {
    }

    @Test
    void getFeatureImportance() {
    }
}