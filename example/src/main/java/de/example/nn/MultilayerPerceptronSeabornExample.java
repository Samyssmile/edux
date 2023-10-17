package de.example.nn;


import de.edux.api.Classifier;
import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.nn.network.api.Dataset;
import de.example.data.seaborn.Penguin;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import de.example.data.seaborn.SeabornDataProcessor;
import de.example.data.seaborn.SeabornProvider;

import java.io.File;
import java.util.List;

public class MultilayerPerceptronSeabornExample {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;
    private static final boolean FILTER_INCOMPLETE_RECORDS = true;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.75;
    private static final File CSV_FILE = new File("example" + File.separator + "datasets" + File.separator + "seaborn-penguins" + File.separator + "penguins.csv");

    public static void main(String[] args) {
        var seabornDataProcessor = new SeabornDataProcessor();
        List<Penguin> data = seabornDataProcessor.loadDataSetFromCSV(CSV_FILE, ',', SHUFFLE, NORMALIZE, FILTER_INCOMPLETE_RECORDS);

        Dataset<Penguin> dataset = seabornDataProcessor.split(data, TRAIN_TEST_SPLIT_RATIO);
        var seabornProvider = new SeabornProvider(data, dataset.trainData(), dataset.testData());
        seabornProvider.printStatistics();
        double[][] features = seabornProvider.getTrainFeatures();
        double[][] labels = seabornProvider.getTrainLabels();

        double[][] testFeatures = seabornProvider.getTestFeatures();
        double[][] testLabels = seabornProvider.getTestLabels();

        NetworkConfiguration networkConfiguration = new NetworkConfiguration(features[0].length, List.of(32, 6), 3, 0.01, 1000, ActivationFunction.LEAKY_RELU, ActivationFunction.SOFTMAX, LossFunction.CATEGORICAL_CROSS_ENTROPY, Initialization.XAVIER, Initialization.XAVIER);
        Classifier multilayerPerceptron = new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);
        multilayerPerceptron.train(features, labels);
        multilayerPerceptron.evaluate(testFeatures, testLabels);
    }

}
