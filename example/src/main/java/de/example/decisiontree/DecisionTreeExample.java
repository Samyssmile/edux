package de.example.decisiontree;

import de.edux.ml.decisiontree.DecisionTree;
import de.edux.ml.decisiontree.IDecisionTree;
import de.example.data.IrisProvider;
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

    // 1 - SATOSA 2 - VERSICOLOR 3 - VIRGINICA
    double[] decisionTreeTrainLabels = convert2DLabelArrayTo1DLabelArray(labels);

    // Train Decision Tree
    IDecisionTree decisionTree = new DecisionTree();
    decisionTree.train(features, decisionTreeTrainLabels, 6, 2, 1, 4);

    // Evaluate Decision Tree
    double[][] testFeatures = datasetProvider.getTestFeatures();
    double[][] testLabels = datasetProvider.getTestLabels();
    double[] decisionTreeTestLabels = convert2DLabelArrayTo1DLabelArray(testLabels);
    decisionTree.evaluate(testFeatures, decisionTreeTestLabels);

    // Get Feature Importance
    double[] featureImportance = decisionTree.getFeatureImportance();
    System.out.println("Feature Importance: " + Arrays.toString(featureImportance));

  }

  private static double[] convert2DLabelArrayTo1DLabelArray(double[][] labels) {
    double[] decisionTreeTrainLabels = new double[labels.length];
    for (int i = 0; i < labels.length; i++) {
      for (int j = 0; j < labels[i].length; j++) {
        if (labels[i][j] == 1) {
          decisionTreeTrainLabels[i] = (j+1);
        }
      }
    }
    return decisionTreeTrainLabels;
  }
}
