package de.edux.ml.decisiontree;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A decision tree classifier.
 * <p>
 * This class implements a binary decision tree algorithm for classification.
 * The decision tree is built by recursively splitting the training data based on
 * the feature that results in the minimum Gini index, which is a measure of impurity.
 * </p>
 *
 * <p>
 * Once the decision tree is built, new instances can be classified by traversing the tree
 * from the root to a leaf node. The class of the leaf node is then assigned to the instance.
 * </p>
 *
 * <p>
 * The decision tree algorithm implemented here includes several stopping conditions to avoid
 * overfitting, including a maximum depth, a minimum number of samples per leaf, and a minimum
 * number of samples to allow a split.
 * </p>
 *
 * <p>
 * The decision tree can be used for multiclass classification problems. For binary classification,
 * the output is either 0 or 1. For multiclass classification, the output is the class with the
 * maximum frequency in the leaf node.
 * </p>
 */
public class DecisionTree implements IDecisionTree {
  private static final Logger LOG = LoggerFactory.getLogger(DecisionTree.class);
  private Node root;
  private int maxDepth;
  private int minSamplesSplit;
  private int minSamplesLeaf;
  private int maxLeafNodes;



  private double calculateGiniIndex(double[] labels) {
    if (labels.length == 0) {
      return 0.0;
    }

    int[] counts = new int[(int) Arrays.stream(labels).max().getAsDouble() + 1];
    Arrays.stream(labels).forEach(label -> counts[(int) label]++);
    return 1.0
        - Arrays.stream(counts)
            .mapToDouble(count -> Math.pow((double) count / labels.length, 2))
            .sum();
  }

  private double calculateSplitGiniIndex(double[][] leftData, double[][] rightData) {
    double[] leftLabels = Arrays.stream(leftData).mapToDouble(row -> row[row.length - 1]).toArray();
    double[] rightLabels =
        Arrays.stream(rightData).mapToDouble(row -> row[row.length - 1]).toArray();
    double leftGiniIndex = calculateGiniIndex(leftLabels);
    double rightGiniIndex = calculateGiniIndex(rightLabels);
    return leftData.length * leftGiniIndex / (leftData.length + rightData.length)
        + rightData.length * rightGiniIndex / (leftData.length + rightData.length);
  }

  private double[][] splitData(double[][] data, int column, double value) {
    return Arrays.stream(data).filter(row -> row[column] < value).toArray(double[][]::new);
  }

  private double[][] getFeatures(double[][] data) {
    return Arrays.stream(data)
        .map(row -> Arrays.copyOf(row, row.length - 1))
        .toArray(double[][]::new);
  }

  private void buildTree(Node node) {
    if (node.data.length <= minSamplesSplit
        || getDepth(node) >= maxDepth
        || getLeafCount(node) >= maxLeafNodes) {
      node.isLeaf = true;
      return;
    }

    double minGiniIndex = Double.MAX_VALUE;
    int minColumn = -1;
    double minValue = Double.MAX_VALUE;
    double[][] leftData = new double[0][];
    double[][] rightData = new double[0][];

    for (int column = 0; column < getFeatures(node.data)[0].length; column++) {
      for (double[] row : node.data) {
        double[][] leftSplitData = splitData(node.data, column, row[column], true);
        double[][] rightSplitData = splitData(node.data, column, row[column], false);
        double giniIndex = calculateSplitGiniIndex(leftSplitData, rightSplitData);

        if (giniIndex < minGiniIndex && leftSplitData.length >= minSamplesLeaf && rightSplitData.length >= minSamplesLeaf) {
          minGiniIndex = giniIndex;
          minColumn = column;
          minValue = row[column];
          leftData = leftSplitData;
          rightData = rightSplitData;
        }
      }
    }


    if (minColumn == -1) {
      node.isLeaf = true;
    } else {
      node.splitFeature = minColumn;
      node.value = minValue;
      node.left = new Node(leftData);
      node.right = new Node(rightData);
      buildTree(node.left);
      buildTree(node.right);
    }
  }

  @Override
  public void train(
      double[][] features,
      int[] labels,
      int maxDepth,
      int minSamplesSplit,
      int minSamplesLeaf,
      int maxLeafNodes) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.maxLeafNodes = maxLeafNodes;

    double[][] data = new double[features.length][];
    for (int i = 0; i < features.length; i++) {
      data[i] = Arrays.copyOf(features[i], features[i].length + 1);
      data[i][data[i].length - 1] = labels[i];
    }
    root = new Node(data);
    buildTree(root);
  }

  @Override
  // Add to the DecisionTree class
  public double predict(double[] feature) {
    return predict(feature, root);
  }

  private double predict(double[] feature, Node node) {
    // If we are at a leaf node, return the most common label
    if (node.isLeaf) {
      return getMostCommonLabel(node.data);
    }

    // Else move to the next node
    if (feature[node.splitFeature] < node.value) {
      return predict(feature, node.left);
    } else {
      return predict(feature, node.right);
    }
  }

  private double getMostCommonLabel(double[][] data) {
    return Arrays.stream(data)
        .mapToDouble(row -> row[row.length - 1])
        .boxed()
        .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
        .entrySet()
        .stream()
        .max(Map.Entry.comparingByValue())
        .get()
        .getKey();
  }

  @Override
  public double evaluate(double[][] features, int[] labels) {
    int correctPredictions = 0;
    for (int i = 0; i < features.length; i++) {
      if (predict(features[i]) == labels[i]) {
        correctPredictions++;
      }
    }
    double accuracy = (double) correctPredictions / features.length;
    LOG.info("Accuracy: " + String.format("%.4f", accuracy * 100) + "%");
    return accuracy;
  }

  @Override
  public double[] getFeatureImportance() {
    int numFeatures = root.data[0].length - 1;
    double[] importances = new double[numFeatures];
    calculateFeatureImportance(root, importances);
    return importances;
  }

  private double calculateFeatureImportance(Node node, double[] importances) {
    if (node == null || node.isLeaf) {
      return 0;
    }

    double importance = calculateGiniIndex(getLabels(node.data))
            - calculateSplitGiniIndex(node.left.data, node.right.data);
    importances[node.splitFeature] += importance;

    return importance + calculateFeatureImportance(node.left, importances) + calculateFeatureImportance(node.right, importances);
  }

  private double[] getLabels(double[][] data) {
    return Arrays.stream(data)
        .mapToDouble(row -> row[row.length - 1])
        .toArray();
  }


  private int getDepth(Node node) {
    if (node == null) {
      return 0;
    }

    return Math.max(getDepth(node.left), getDepth(node.right)) + 1;
  }

  private double[][] splitData(double[][] data, int column, double value, boolean isLeftSplit) {
    if (isLeftSplit) {
      return Arrays.stream(data).filter(row -> row[column] < value).toArray(double[][]::new);
    } else {
      return Arrays.stream(data).filter(row -> row[column] >= value).toArray(double[][]::new);
    }
  }

  private int getLeafCount(Node node) {
    if (node == null) {
      return 0;
    } else if (node.isLeaf) {
      return 1;
    } else {
      return getLeafCount(node.left) + getLeafCount(node.right);
    }
  }
}
