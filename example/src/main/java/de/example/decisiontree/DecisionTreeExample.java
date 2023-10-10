package de.example.decisiontree;

import de.edux.ml.decisiontree.DecisionTree;
import de.example.data.iris.IrisProvider;

import java.util.Arrays;

public class DecisionTreeExample {
  private static final boolean SHUFFLE = true;
  private static final boolean NORMALIZE = true;

  public static void main(String[] args) {
    // Get IRIS dataset
    var datasetProvider = new IrisProvider(NORMALIZE, SHUFFLE, 0.6);
    datasetProvider.printStatistics();

    //Get Features and Labels
    double[][] features = datasetProvider.getTrainFeatures();
    double[][] labels = datasetProvider.getTrainLabels();

    // Train Decision Tree
    DecisionTree decisionTree = new DecisionTree(8, 2, 1, 4);
    decisionTree.train(features, labels);

    // Evaluate Decision Tree
    double[][] testFeatures = datasetProvider.getTestFeatures();
    double[][] testLabels = datasetProvider.getTestLabels();
    decisionTree.evaluate(testFeatures, testLabels);

  }
}
