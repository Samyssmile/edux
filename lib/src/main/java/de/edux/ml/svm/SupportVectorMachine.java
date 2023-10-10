package de.edux.ml.svm;

import de.edux.api.Classifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * The {@code SupportVectorMachine} class is an implementation of a Support Vector Machine (SVM) classifier, utilizing the one-vs-one strategy for multi-class classification.
 * This SVM implementation accepts a kernel function and trains separate binary classifiers for each pair of classes in the training set, using provided kernel function and regularization parameter C.
 * During the prediction, each model in the pair casts a vote and the final predicted class is the one that gets the most votes among all binary classifiers.
 *
 * <p>Example usage:</p>
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
 * <p>Note: Label arrays are expected to be in one-hot encoding format and will be internally converted to single label format for training.</p>
 *
 * @see de.edux.api.Classifier
 */
public class SupportVectorMachine implements Classifier {
    private static final Logger LOG = LoggerFactory.getLogger(SupportVectorMachine.class);
    private final SVMKernel kernel;
    private final double c;
    private final Map<String, SVMModel> models;
    public SupportVectorMachine(SVMKernel kernel, double c) {
        this.kernel = kernel;
        this.c = c;
        this.models = new HashMap<>();
    }

    @Override
    public boolean train(double[][] features, double[][] labels) {
        var oneDLabels = convert2DLabelArrayTo1DLabelArray(labels);
        // Identify unique class labels
        Set<Integer> uniqueLabels = Arrays.stream(oneDLabels).boxed().collect(Collectors.toSet());
        Integer[] uniqueLabelsArray = uniqueLabels.toArray(new Integer[0]);

        // In One-vs-One, you should consider every possible pair of classes
        for (int i = 0; i < uniqueLabelsArray.length; i++) {
            for (int j = i + 1; j < uniqueLabelsArray.length; j++) {
                String key = uniqueLabelsArray[i] + "-" + uniqueLabelsArray[j];
                SVMModel model = new SVMModel(kernel, c);

                // Filter the features and labels for the two classes
                List<double[]> list = new ArrayList<>();
                List<Integer> pairLabelsList = new ArrayList<>();
                for (int k = 0; k < features.length; k++) {
                    if (oneDLabels[k] == uniqueLabelsArray[i] || oneDLabels[k] == uniqueLabelsArray[j]) {
                        list.add(features[k]);
                        // Ensure that the sign of the label matches our assumption
                        pairLabelsList.add(oneDLabels[k] == uniqueLabelsArray[i] ? 1 : -1);
                    }
                }
                double[][] pairFeatures = list.toArray(new double[0][]);
                int[] pairLabels = pairLabelsList.stream().mapToInt(Integer::intValue).toArray();

                // Train the model on the pair
                model.train(pairFeatures, pairLabels);
                models.put(key, model);
            }
        }
        return true;
    }

    @Override
    public double[] predict(double[] features) {
        Map<Integer, Integer> voteCount = new HashMap<>();
        // In One-vs-One, you look at the prediction of each model and count the votes
        for (Map.Entry<String, SVMModel> entry : models.entrySet()) {
            int prediction = entry.getValue().predict(features);

            // map prediction back to actual class label
            String[] classes = entry.getKey().split("-");
            int classLabel = (prediction == 1) ? Integer.parseInt(classes[0]) : Integer.parseInt(classes[1]);

            voteCount.put(classLabel, voteCount.getOrDefault(classLabel, 0) + 1);
        }

        // The final prediction is the class with the most votes
        int prediction = voteCount.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        double[] result = new double[models.size()];
        result[prediction - 1] = 1;
        return  result;
    }
    @Override
    public double evaluate(double[][] features, double[][]  labels) {
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
                    decisionTreeTrainLabels[i] = (j+1);
                }
            }
        }
        return decisionTreeTrainLabels;
    }
}

