package de.edux.ml.decisiontree;

import de.edux.api.Classifier;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Decision Tree classifier for predictive modeling.
 *
 * <p>The {@code DecisionTree} class is a binary tree where each node represents a decision on a
 * particular feature from the input feature vector, effectively partitioning the input space into
 * regions with similar output labels. The tree is built recursively by selecting splits that
 * minimize the Gini impurity of the resultant partitions.
 *
 * <p>Features:
 *
 * <ul>
 *   <li>Supports binary classification problems.
 *   <li>Utilizes the Gini impurity to determine optimal feature splits.
 *   <li>Enables control over tree depth and complexity through various hyperparameters.
 * </ul>
 *
 * <p>Hyperparameters include:
 *
 * <ul>
 *   <li>{@code maxDepth}: The maximum depth of the tree.
 *   <li>{@code minSamplesSplit}: The minimum number of samples required to split an internal node.
 *   <li>{@code minSamplesLeaf}: The minimum number of samples required to be at a leaf node.
 *   <li>{@code maxLeafNodes}: The maximum number of leaf nodes in the tree.
 * </ul>
 *
 * <p>Usage example:
 *
 * <pre>{@code
 * DecisionTree classifier = new DecisionTree(10, 2, 1, 50);
 * classifier.train(trainingFeatures, trainingLabels);
 * double accuracy = classifier.evaluate(testFeatures, testLabels);
 * }</pre>
 *
 * <p>Note: This class requires a thorough validation of input data and parameters, ensuring they
 * are never {@code null}, have appropriate dimensions, and adhere to any other prerequisites or
 * assumptions, to guarantee robustness and avoid runtime exceptions.
 *
 * @see Classifier
 */
public class DecisionTree implements Classifier {
  private static final Logger LOG = LoggerFactory.getLogger(DecisionTree.class);
  private final int maxDepth;
  private final int minSamplesSplit;
  private final int minSamplesLeaf;
  private final int maxLeafNodes;
  private final Map<Integer, Double> featureImportances;
  private Node root;
  private int currentLeafNodes;

  public DecisionTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf, int maxLeafNodes) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.maxLeafNodes = maxLeafNodes;
    this.currentLeafNodes = 0;
    this.featureImportances = new HashMap<>();
  }

  @Override
  public boolean train(double[][] features, double[][] labels) {
    try {
      if (features == null
          || labels == null
          || features.length == 0
          || labels.length == 0
          || features.length != labels.length) {
        LOG.error("Invalid training data");
        return false;
      }

      this.root = buildTree(features, labels, 0);

      return true;
    } catch (Exception e) {
      LOG.error("An error occurred during training", e);
      return false;
    }
  }

  private Node buildTree(double[][] features, double[][] labels, int depth) {
    Node node = new Node(features);
    node.predictedLabel = getMajorityLabel(labels);

    if (shouldTerminate(features, depth)) {
      currentLeafNodes++;
      return node;
    }

    SplitResult bestSplit = findBestSplit(features, labels);
    if (bestSplit != null) {
      applyBestSplit(node, bestSplit, features, labels, depth);
    } else {
      currentLeafNodes++;
    }

    return node;
  }

  private boolean shouldTerminate(double[][] features, int depth) {
    boolean maxDepthReached = depth >= maxDepth;
    boolean tooFewSamples = features.length < minSamplesSplit;
    boolean maxLeafNodesReached = currentLeafNodes >= maxLeafNodes;

    if (maxDepthReached || tooFewSamples || maxLeafNodesReached) {
      return true;
    }
    return false;
  }

  private SplitResult findBestSplit(double[][] features, double[][] labels) {
    double bestGini = Double.MAX_VALUE;
    SplitResult bestSplit = null;

    for (int featureIndex = 0; featureIndex < features[0].length; featureIndex++) {
      for (double[] feature : features) {
        double[][] leftFeatures = filterRows(features, featureIndex, feature[featureIndex], true);
        double[][] rightFeatures = filterRows(features, featureIndex, feature[featureIndex], false);

        double[][] leftLabels = filterRows(labels, leftFeatures, features);
        double[][] rightLabels = filterRows(labels, rightFeatures, features);

        double gini = computeGini(leftLabels, rightLabels);

        if (gini < bestGini) {
          bestGini = gini;
          updateFeatureImportances(featureIndex, gini);
          bestSplit =
              new SplitResult(
                  featureIndex,
                  feature[featureIndex],
                  leftFeatures,
                  rightFeatures,
                  leftLabels,
                  rightLabels);
        }
      }
    }

    return bestSplit;
  }

  private void applyBestSplit(
      Node node, SplitResult bestSplit, double[][] features, double[][] labels, int depth) {
    node.splitFeatureIndex = bestSplit.featureIndex;
    node.splitValue = bestSplit.splitValue;

    if (bestSplit.bestLeftFeatures != null
        && bestSplit.bestRightFeatures != null
        && bestSplit.bestLeftFeatures.length >= minSamplesLeaf
        && bestSplit.bestRightFeatures.length >= minSamplesLeaf) {

      if (currentLeafNodes + 2 <= maxLeafNodes) {
        node.left = buildTree(bestSplit.bestLeftFeatures, bestSplit.bestLeftLabels, depth + 1);
        node.right = buildTree(bestSplit.bestRightFeatures, bestSplit.bestRightLabels, depth + 1);
        currentLeafNodes += 2;
      } else {
        currentLeafNodes++;
      }
    } else {
      currentLeafNodes++;
    }
  }

  private void updateFeatureImportances(int featureIndex, double giniReduction) {
    featureImportances.merge(featureIndex, giniReduction, Double::sum);
  }

  public Map<Integer, Double> getFeatureImportances() {
    double totalImportance =
        featureImportances.values().stream().mapToDouble(Double::doubleValue).sum();
    return featureImportances.entrySet().stream()
        .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue() / totalImportance));
  }

  private double[][] filterRows(
      double[][] matrix, int featureIndex, double value, boolean lessThan) {
    return Arrays.stream(matrix)
        .filter(
            row ->
                (lessThan && row[featureIndex] < value)
                    || (!lessThan && row[featureIndex] >= value))
        .toArray(double[][]::new);
  }

  private double[][] filterRows(
      double[][] labels, double[][] filteredFeatures, double[][] originalFeatures) {
    List<double[]> filteredLabelsList = new ArrayList<>();
    for (double[] filteredFeature : filteredFeatures) {
      for (int i = 0; i < originalFeatures.length; i++) {
        if (Arrays.equals(filteredFeature, originalFeatures[i])) {
          filteredLabelsList.add(labels[i]);
          break;
        }
      }
    }
    return filteredLabelsList.toArray(new double[0][0]);
  }

  private double[] getMajorityLabel(double[][] labels) {
    return Arrays.stream(labels)
        .map(Arrays::toString) // Convert double[] to String for grouping
        .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
        .entrySet()
        .stream()
        .max(Map.Entry.comparingByValue())
        .map(Map.Entry::getKey)
        .map(
            str ->
                Arrays.stream(str.substring(1, str.length() - 1).split(", "))
                    .mapToDouble(Double::parseDouble)
                    .toArray()) // Convert String back to double[]
        .orElseThrow(RuntimeException::new);
  }

  @Override
  public double evaluate(double[][] testInputs, double[][] testTargets) {
    if (testInputs == null
        || testTargets == null
        || testInputs.length == 0
        || testTargets.length == 0
        || testInputs.length != testTargets.length) {
      LOG.error("Invalid test data");
      return 0;
    }

    long correctPredictions = 0;
    for (int i = 0; i < testInputs.length; i++) {
      double[] prediction = predict(testInputs[i]);
      if (Arrays.equals(prediction, testTargets[i])) {
        correctPredictions++;
      }
    }

    double accuracy = (double) correctPredictions / testInputs.length;
    LOG.info(String.format("Decision Tree -  accuracy: %.2f%%", accuracy * 100));
    return accuracy;
  }

  @Override
  public double[] predict(double[] feature) {
    return predictRecursive(root, feature);
  }

  private double[] predictRecursive(Node node, double[] feature) {
    if (node == null || feature == null) {
      throw new IllegalArgumentException("Node and feature cannot be null");
    }

    if (node.left == null && node.right == null) {
      return node.predictedLabel;
    }

    if (node.splitFeatureIndex >= feature.length) {
      throw new IllegalArgumentException("splitFeatureIndex is out of bounds of feature array");
    }

    if (feature[node.splitFeatureIndex] < node.splitValue) {
      if (node.left == null) {
        throw new IllegalStateException("Left node is null when trying to traverse left");
      }
      return predictRecursive(node.left, feature);
    } else {
      if (node.right == null) {
        throw new IllegalStateException("Right node is null when trying to traverse right");
      }
      return predictRecursive(node.right, feature);
    }
  }

  private double computeGini(double[][] leftLabels, double[][] rightLabels) {
    double leftImpurity = computeImpurity(leftLabels);
    double rightImpurity = computeImpurity(rightLabels);
    double leftWeight = ((double) leftLabels.length) / (leftLabels.length + rightLabels.length);
    double rightWeight = ((double) rightLabels.length) / (leftLabels.length + rightLabels.length);
    return leftWeight * leftImpurity + rightWeight * rightImpurity;
  }

  private double computeImpurity(double[][] labels) {
    double impurity = 1.0;
    Map<String, Long> labelCounts =
        Arrays.stream(labels)
            .map(Arrays::toString)
            .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
    for (Long count : labelCounts.values()) {
      double p = ((double) count) / labels.length;
      impurity -= p * p;
    }
    return impurity;
  }

  private static class SplitResult {
    int featureIndex;
    double splitValue;
    double[][] bestLeftFeatures;
    double[][] bestRightFeatures;
    double[][] bestLeftLabels;
    double[][] bestRightLabels;

    SplitResult(
        int featureIndex,
        double splitValue,
        double[][] bestLeftFeatures,
        double[][] bestRightFeatures,
        double[][] bestLeftLabels,
        double[][] bestRightLabels) {
      this.featureIndex = featureIndex;
      this.splitValue = splitValue;
      this.bestLeftFeatures = bestLeftFeatures;
      this.bestRightFeatures = bestRightFeatures;
      this.bestLeftLabels = bestLeftLabels;
      this.bestRightLabels = bestRightLabels;
    }
  }

  private static class Node {
    double[][] data;
    Node left;
    Node right;
    int splitFeatureIndex;
    double splitValue;
    double[] predictedLabel;

    public Node(double[][] data) {
      this.data = data;
    }
  }
}
