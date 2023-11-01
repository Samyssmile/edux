package de.example.benchmark;

import de.edux.api.Classifier;
import de.edux.data.provider.DataProcessor;
import de.edux.data.reader.CSVIDataReader;
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
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Compare the performance of different classifiers */
public class Benchmark {
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
  Map<String, List<Double>> results = new ConcurrentHashMap<>();
  private double[][] trainFeatures;
  private double[][] trainLabels;
  private double[][] testFeatures;
  private double[][] testLabels;
  private MultilayerPerceptron multilayerPerceptron;
  private NetworkConfiguration networkConfiguration;
  private DataProcessor dataProcessor;

  public Benchmark() {
    results.put("KNN", new ArrayList<>());
    results.put("DecisionTree", new ArrayList<>());
    results.put("RandomForest", new ArrayList<>());
    results.put("SVM", new ArrayList<>());
    results.put("MLP", new ArrayList<>());

    init();
  }

  public static void main(String[] args) {
    new Benchmark().run();
  }

  private void init() {
    var featureColumnIndices = new int[] {0, 1, 2, 3};
    var targetColumnIndex = 4;

    dataProcessor =
        new DataProcessor(new CSVIDataReader())
            .loadDataSetFromCSV(CSV_FILE, ',', SKIP_HEAD, featureColumnIndices, targetColumnIndex)
            .normalize()
            .shuffle()
            .split(TRAIN_TEST_SPLIT_RATIO);

    trainFeatures = dataProcessor.getTrainFeatures(featureColumnIndices);
    trainLabels = dataProcessor.getTrainLabels(targetColumnIndex);
    testFeatures = dataProcessor.getTestFeatures(featureColumnIndices);
    testLabels = dataProcessor.getTestLabels(targetColumnIndex);
  }

  private void run() {

    for (int run = 0; run < 10; run++) {
      Classifier knn = new KnnClassifier(2);
      Classifier decisionTree = new DecisionTree(2, 2, 3, 12);
      Classifier randomForest = new RandomForest(500, 10, 2, 3, 3, 60);
      Classifier svm = new SupportVectorMachine(SVMKernel.LINEAR, 1);

      networkConfiguration =
          new NetworkConfiguration(
              trainFeatures[0].length,
              List.of(64, 256, 512),
              3,
              0.01,
              300,
              ActivationFunction.LEAKY_RELU,
              ActivationFunction.SOFTMAX,
              LossFunction.CATEGORICAL_CROSS_ENTROPY,
              Initialization.XAVIER,
              Initialization.XAVIER);
      multilayerPerceptron =
          new MultilayerPerceptron(networkConfiguration, testFeatures, testLabels);

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
      init();
    }

    System.out.println("Classifier performances (sorted by average accuracy):");
    results.entrySet().stream()
        .map(
            entry -> {
              double avgAccuracy =
                  entry.getValue().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
              return Map.entry(entry.getKey(), avgAccuracy);
            })
        .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
        .forEachOrdered(
            entry -> {
              System.out.printf("%s: %.2f%%\n", entry.getKey(), entry.getValue() * 100);
            });

    System.out.println("\nClassifier best and worst performances:");
    results.forEach(
        (classifierName, accuracies) -> {
          double maxAccuracy =
              accuracies.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
          double minAccuracy =
              accuracies.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
          System.out.printf(
              "%s: Best: %.2f%%, Worst: %.2f%%\n",
              classifierName, maxAccuracy * 100, minAccuracy * 100);
        });
  }
}
