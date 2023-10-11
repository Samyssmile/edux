package de.example.decisiontree;

import de.edux.ml.decisiontree.DecisionTree;
import de.example.data.iris.IrisProvider;

import java.util.Arrays;

public class DecisionTreeExample {
  private static final boolean SHUFFLE = true;
  private static final boolean NORMALIZE = true;

  public static void main(String[] args) {

    var datasetProvider = new IrisProvider(NORMALIZE, SHUFFLE, 0.6);
    datasetProvider.printStatistics();

    double[][] features = datasetProvider.getTrainFeatures();
    double[][] labels = datasetProvider.getTrainLabels();

    DecisionTree decisionTree = new DecisionTree(8, 2, 1, 4);
    decisionTree.train(features, labels);

    double[][] testFeatures = datasetProvider.getTestFeatures();
    double[][] testLabels = datasetProvider.getTestLabels();
    decisionTree.evaluate(testFeatures, testLabels);

  }
}
