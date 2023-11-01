package de.edux.ml.svm;

import de.edux.api.Classifier;
import java.util.*;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code SupportVectorMachine} class is an implementation of a Support Vector Machine (SVM)
 * classifier, utilizing the one-vs-one strategy for multi-class classification. This SVM
 * implementation accepts a kernel function and trains separate binary classifiers for each pair of
 * classes in the training set, using provided kernel function and regularization parameter C.
 * During the prediction, each model in the pair casts a vote and the final predicted class is the
 * one that gets the most votes among all binary classifiers.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * SVMKernel kernel = ... ;  // Define an appropriate SVM kernel function
 * double c = ... ;  // Define an appropriate regularization parameter
 *
 * SupportVectorMachine svm = new SupportVectorMachine(kernel, c);
 * svm.train(trainingFeatures, trainingLabels);
 *
 * double[] prediction = svm.predict(inputFeatures);
 * double accuracy = svm.evaluate(testFeatures, testLabels);
 * }</pre>
 *
 * <p>Note: Label arrays are expected to be in one-hot encoding format and will be internally
 * converted to single label format for training.
 *
 * @see de.edux.api.Classifier
 */
public class SupportVectorMachine implements Classifier {
  private static final Logger LOG = LoggerFactory.getLogger(SupportVectorMachine.class);
  private final SVMKernel kernel;
  private final double c;
  private final Map<String, SVMModel> models;

  /**
   * Constructs a new instance of SupportVectorMachine with a specified kernel and regularization
   * parameter.
   *
   * <p>This constructor initializes a new Support Vector Machine (SVM) for classification tasks.
   * The SVM employs a one-vs-one strategy for multi-class classification. Each model pair within
   * the SVM is trained using the provided kernel function and the regularization parameter C.
   *
   * <p>The kernel is crucial for handling non-linearly separable data by defining a new space in
   * which data points are projected. The correct choice of a kernel significantly impacts the
   * performance of the SVM. The regularization parameter C controls the trade-off between achieving
   * a low training error and a low testing error that is the ability of the SVM to generalize to
   * unseen data.
   *
   * @param kernel The kernel to be used for the transformation of the input space. This is
   *     necessary for achieving an optimal separation in a higher-dimensional space when data is
   *     not linearly separable in the original space. The kernel defines how data points in space
   *     are interpreted based on their similarity.
   * @param c The regularization parameter that controls the trade-off between allowing training
   *     errors and enforcing rigid margins. It helps to prevent overfitting by controlling the
   *     strength of the penalty for errors. A higher value of C tries to minimize the
   *     classification error, potentially at the expense of simplicity, while a lower value of C
   *     prioritizes simplicity, potentially allowing some misclassifications.
   */
  public SupportVectorMachine(SVMKernel kernel, double c) {
    this.kernel = kernel;
    this.c = c;
    this.models = new HashMap<>();
  }

  @Override
  public boolean train(double[][] features, double[][] labels) {
    var oneDLabels = convert2DLabelArrayTo1DLabelArray(labels);
    Set<Integer> uniqueLabels = Arrays.stream(oneDLabels).boxed().collect(Collectors.toSet());
    Integer[] uniqueLabelsArray = uniqueLabels.toArray(new Integer[0]);

    for (int i = 0; i < uniqueLabelsArray.length; i++) {
      for (int j = i + 1; j < uniqueLabelsArray.length; j++) {
        String key = uniqueLabelsArray[i] + "-" + uniqueLabelsArray[j];
        SVMModel model = new SVMModel(kernel, c);

        List<double[]> list = new ArrayList<>();
        List<Integer> pairLabelsList = new ArrayList<>();
        for (int k = 0; k < features.length; k++) {
          if (oneDLabels[k] == uniqueLabelsArray[i] || oneDLabels[k] == uniqueLabelsArray[j]) {
            list.add(features[k]);
            pairLabelsList.add(oneDLabels[k] == uniqueLabelsArray[i] ? 1 : -1);
          }
        }
        double[][] pairFeatures = list.toArray(new double[0][]);
        int[] pairLabels = pairLabelsList.stream().mapToInt(Integer::intValue).toArray();

        model.train(pairFeatures, pairLabels);
        models.put(key, model);
      }
    }
    return true;
  }

  @Override
  public double[] predict(double[] features) {
    Map<Integer, Integer> voteCount = new HashMap<>();

    for (Map.Entry<String, SVMModel> entry : models.entrySet()) {
      int prediction = entry.getValue().predict(features);

      String[] classes = entry.getKey().split("-");
      int classLabel =
          (prediction == 1) ? Integer.parseInt(classes[0]) : Integer.parseInt(classes[1]);

      voteCount.put(classLabel, voteCount.getOrDefault(classLabel, 0) + 1);
    }

    int prediction = voteCount.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    double[] result = new double[models.size()];
    result[prediction - 1] = 1;
    return result;
  }

  @Override
  public double evaluate(double[][] features, double[][] labels) {
    int correct = 0;
    for (int i = 0; i < features.length; i++) {
      boolean match = Arrays.equals(predict(features[i]), labels[i]);
      if (match) {
        correct++;
      }
    }

    double accuracy = (double) correct / features.length;

    LOG.info("SVM - Accuracy: " + String.format("%.4f", accuracy * 100) + "%");
    return accuracy;
  }

  private int[] convert2DLabelArrayTo1DLabelArray(double[][] labels) {
    int[] decisionTreeTrainLabels = new int[labels.length];
    for (int i = 0; i < labels.length; i++) {
      for (int j = 0; j < labels[i].length; j++) {
        if (labels[i][j] == 1) {
          decisionTreeTrainLabels[i] = (j + 1);
        }
      }
    }
    return decisionTreeTrainLabels;
  }
}
