package de.edux.ml.nn.network;

import de.edux.data.handler.EIncompleteRecordsHandlerStrategy;
import de.edux.data.provider.SeabornDataProcessor;
import de.edux.data.provider.SeabornProvider;
import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.RepeatedTest;

import java.io.File;
import java.net.URL;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MultilayerPerceptronTest {
    private static final boolean SHUFFLE = true;
    private static final boolean SKIP_HEADLINE = true;
    private static final EIncompleteRecordsHandlerStrategy INCOMPLETE_RECORD_HANDLER_STRATEGY = EIncompleteRecordsHandlerStrategy.DROP_RECORDS;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.7;
    private static final String CSV_FILE_PATH = "testdatasets/seaborn-penguins/penguins.csv";
    private SeabornProvider seabornProvider;

    @BeforeEach
    void setUp() {
        URL url = MultilayerPerceptronTest.class.getClassLoader().getResource(CSV_FILE_PATH);
        if (url == null) {
            throw new IllegalStateException("Cannot find file: " + CSV_FILE_PATH);
        }
        File csvFile = new File(url.getPath());
        var seabornDataProcessor = new SeabornDataProcessor();
        var dataset = seabornDataProcessor.loadDataSetFromCSV(csvFile, ',', true, true, INCOMPLETE_RECORD_HANDLER_STRATEGY);
        seabornDataProcessor.normalize(dataset);
        var trainTestSplittedList = seabornDataProcessor.split(dataset, TRAIN_TEST_SPLIT_RATIO);
        seabornProvider = new SeabornProvider(dataset, trainTestSplittedList.trainData(), trainTestSplittedList.testData());
        System.out.println("SeabornProvider loaded");
    }

    @RepeatedTest(3)
    void shouldReachModelAccuracyAtLeast70() {
        double[][] features = seabornProvider.getTrainFeatures();
        double[][] labels = seabornProvider.getTrainLabels();

        double[][] testFeatures = seabornProvider.getTestFeatures();
        double[][] testLabels = seabornProvider.getTestLabels();

        assertTrue(features.length > 0);
        assertTrue(labels.length > 0);
        assertTrue(testFeatures.length > 0);
        assertTrue(testLabels.length > 0);

        NetworkConfiguration networkConfiguration = new NetworkConfiguration(features[0].length, List.of(128,256, 512), 3, 0.01, 300, ActivationFunction.LEAKY_RELU, ActivationFunction.SOFTMAX, LossFunction.CATEGORICAL_CROSS_ENTROPY, Initialization.XAVIER, Initialization.XAVIER);

        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);
        multilayerPerceptron.train(features, labels);
        double accuracy = multilayerPerceptron.evaluate(testFeatures, testLabels);
        assertTrue(accuracy > 0.7);

    }
}