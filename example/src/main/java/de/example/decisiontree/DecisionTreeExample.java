package de.example.decisiontree;

import de.edux.ml.decisiontree.DecisionTree;
import de.edux.ml.decisiontree.IDecisionTree;
import de.example.data.iris.IrisProvider;
import java.util.Arrays;

import static de.edux.util.LabelDimensionConverter.convert2DLabelArrayTo1DLabelArray;

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
    IDecisionTree decisionTree = new DecisionTree();
    decisionTree.train(features, labels, 6, 2, 1, 4);

    // Evaluate Decision Tree
    double[][] testFeatures = datasetProvider.getTestFeatures();
    double[][] testLabels = datasetProvider.getTestLabels();
    decisionTree.evaluate(testFeatures, testLabels);

    // Get Feature Importance
    double[] featureImportance = decisionTree.getFeatureImportance();
    System.out.println("Feature Importance: " + Arrays.toString(featureImportance));
  }

}
