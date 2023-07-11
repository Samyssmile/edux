package de.edux.ml.decisiontree;

public interface IDecisionTree {

  void train(
      double[][] features,
      double[] labels,
      int maxDepth,
      int minSamplesSplit,
      int minSamplesLeaf,
      int maxLeafNodes);

  /**
   * Pedicts the label for the given features.
   *
   * @param features the features to predict
   * @return the predicted label
   */
  double predict(double[] features);
  /**
   * Evaluates the given features and labels against the decision tree.
   *
   * @param features the features to evaluate
   * @param labels the labels to evaluate
   * @return true if the decision tree correctly classified the features and labels, false otherwise
   */
  double evaluate(double[][] features, double[] labels);

  /**
   * Returns the feature importance of the decision tree.
   *
   * @return the feature importance of the decision tree
   */
  double[] getFeatureImportance();
}
