package de.edux.ml.decisiontree;

public interface IDecisionTree {

  /**
   * Returns the feature importance of the decision tree.
   *
   * @return the feature importance of the decision tree
   */
  double[] getFeatureImportance();
}
