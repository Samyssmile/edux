package de.example.benchmark;

import de.edux.api.Classifier;
import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import de.edux.ml.decisiontree.DecisionTree;
import de.edux.ml.knn.KnnClassifier;
import de.edux.ml.nn.config.NetworkConfiguration;
import de.edux.ml.nn.network.MultilayerPerceptron;
import de.edux.ml.randomforest.RandomForest;
import de.edux.ml.svm.SVMKernel;
import de.edux.ml.svm.SupportVectorMachine;
import de.example.data.seaborn.Penguin;
import de.example.data.seaborn.SeabornDataProcessor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

/**
 * Compare the performance of different classifiers
 */
public class Benchmark {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;
    private static final boolean FILTER_INCOMPLETE_RECORDS = true;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.75;
    private static final File CSV_FILE = new File("example" + File.separator + "datasets" + File.separator + "seaborn-penguins" + File.separator + "penguins.csv");
    private double[][] trainFeatures;
    private double[][] trainLabels;
    private double[][] testFeatures;
    private double[][] testLabels;
    private MultilayerPerceptron multilayerPerceptron;
    private NetworkConfiguration networkConfiguration;

    public static void main(String[] args) {
        new Benchmark().run();
    }

    private void run() {
        initFeaturesAndLabels();

        Classifier knn = new KnnClassifier(2);
        Classifier decisionTree = new DecisionTree(8, 2, 1, 3);
        Classifier randomForest = new RandomForest(100, 10, 2, 1, 3, 60);
        Classifier svm = new SupportVectorMachine(SVMKernel.LINEAR, 1);

        networkConfiguration = new NetworkConfiguration(trainFeatures[0].length, List.of(128,256, 512), 3, 0.01, 300, ActivationFunction.LEAKY_RELU, ActivationFunction.SOFTMAX, LossFunction.CATEGORICAL_CROSS_ENTROPY, Initialization.XAVIER, Initialization.XAVIER);
        multilayerPerceptron = new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);
        Map<String, Classifier> classifiers = Map.of(
                "KNN", knn,
                "DecisionTree", decisionTree,
                "RandomForest", randomForest,
                "SVM", svm,
                "MLP", multilayerPerceptron
        );

        Map<String, List<Double>> results = new ConcurrentHashMap<>();
        results.put("KNN", new ArrayList<>());
        results.put("DecisionTree", new ArrayList<>());
        results.put("RandomForest", new ArrayList<>());
        results.put("SVM", new ArrayList<>());
        results.put("MLP", new ArrayList<>());


        IntStream.range(0, 50).forEach(i -> {
            knn.train(trainFeatures, trainLabels);
            decisionTree.train(trainFeatures, trainLabels);
            randomForest.train(trainFeatures, trainLabels);
            svm.train(trainFeatures, trainLabels);
            multilayerPerceptron.train(trainFeatures, trainLabels);

            double knnAccuracy = knn.evaluate(testFeatures, testLabels);
            double decisionTreeAccuracy = decisionTree.evaluate(testFeatures, testLabels);
            double randomForestAccuracy = randomForest.evaluate(testFeatures, testLabels);
            double svmAccuracy = svm.evaluate(testFeatures, testLabels);
            double multilayerPerceptronAccuracy = multilayerPerceptron.evaluate(testFeatures, testLabels);

            results.get("KNN").add(knnAccuracy);
            results.get("DecisionTree").add(decisionTreeAccuracy);
            results.get("RandomForest").add(randomForestAccuracy);
            results.get("SVM").add(svmAccuracy);
            results.get("MLP").add(multilayerPerceptronAccuracy);
            initFeaturesAndLabels();
            updateMLP(testFeatures, testLabels);
        });


        //Sort and print results with numeration begin with best average accuracy
        System.out.println("Classifier performances (sorted by average accuracy):");
        results.entrySet().stream()
                .map(entry -> {
                    double avgAccuracy = entry.getValue().stream()
                            .mapToDouble(Double::doubleValue)
                            .average()
                            .orElse(0.0);
                    return Map.entry(entry.getKey(), avgAccuracy);
                })
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEachOrdered(entry -> {
                    System.out.printf("%s: %.2f%%\n", entry.getKey(), entry.getValue() * 100);
                });

        // Additionally, if you want to show other metrics, such as minimum or maximum accuracy, you can calculate and display them similarly.
        System.out.println("\nClassifier best and worst performances:");
        results.forEach((classifierName, accuracies) -> {
            double maxAccuracy = accuracies.stream()
                    .mapToDouble(Double::doubleValue)
                    .max()
                    .orElse(0.0);
            double minAccuracy = accuracies.stream()
                    .mapToDouble(Double::doubleValue)
                    .min()
                    .orElse(0.0);
            System.out.printf("%s: Best: %.2f%%, Worst: %.2f%%\n", classifierName, maxAccuracy * 100, minAccuracy * 100);
        });


    }

    private void updateMLP(double[][] testFeatures, double[][] testLabels) {
        multilayerPerceptron = new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);
    }

    private void initFeaturesAndLabels() {
        var seabornDataProcessor = new SeabornDataProcessor();
        List<Penguin> data = seabornDataProcessor.loadDataSetFromCSV(CSV_FILE, ',', SHUFFLE, NORMALIZE, FILTER_INCOMPLETE_RECORDS);
        seabornDataProcessor.split(data, TRAIN_TEST_SPLIT_RATIO);

        trainFeatures = seabornDataProcessor.getTrainFeatures();
        trainLabels = seabornDataProcessor.getTrainLabels();

        testFeatures = seabornDataProcessor.getTestFeatures();
        testLabels = seabornDataProcessor.getTestLabels();


    }
}
