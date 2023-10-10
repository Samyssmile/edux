package de.edux.ml.knn;

import de.edux.api.Classifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.PriorityQueue;

/**
 * The {@code KnnClassifier} class provides an implementation of the k-Nearest Neighbors algorithm for classification tasks.
 * It stores the training dataset and predicts the label for new data points based on the majority label of its k-nearest neighbors in the feature space.
 * Distance between data points is computed using the Euclidean distance metric.
 * Optionally, predictions can be weighted by the inverse of the distance to give closer neighbors higher influence.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * int k = 3;  // Specify the number of neighbors to consider
 * KnnClassifier knn = new KnnClassifier(k);
 * knn.train(trainingFeatures, trainingLabels);
 *
 * double[] prediction = knn.predict(inputFeatures);
 * double accuracy = knn.evaluate(testFeatures, testLabels);
 * }</pre>
 *
 * <p>Note: The label arrays should be in one-hot encoding format.</p>
 *
 *
 */
public class KnnClassifier implements Classifier {
    Logger LOG = LoggerFactory.getLogger(KnnClassifier.class);
    private double[][] trainFeatures;
    private double[][] trainLabels;
    private int k;
    private static final double EPSILON = 1e-10;

    /**
     * Initializes a new instance of {@code KnnClassifier} with specified k.
     *
     * @param k an integer value representing the number of neighbors to consider during classification
     * @throws IllegalArgumentException if k is not a positive integer
     */
    public KnnClassifier(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be a positive integer");
        }
        this.k = k;
    }

    @Override
    public boolean train(double[][] features, double[][] labels) {
        if (features.length == 0 || features.length != labels.length) {
            return false;
        }
        this.trainFeatures = features;
        this.trainLabels = labels;
        return true;
    }

    @Override
    public double evaluate(double[][] testInputs, double[][] testTargets) {
        LOG.info("Evaluating...");
        int correct = 0;
        for (int i = 0; i < testInputs.length; i++) {
            double[] prediction = predict(testInputs[i]);
            if (Arrays.equals(prediction, testTargets[i])) {
                correct++;
            }
        }
        double accuracy = (double) correct / testInputs.length;
        LOG.info("KNN - Accuracy: " + accuracy * 100 + "%");
        return accuracy;
    }

    @Override
    public double[] predict(double[] feature) {
        PriorityQueue<Neighbor> pq = new PriorityQueue<>((a, b) -> Double.compare(b.distance, a.distance));
        for (int i = 0; i < trainFeatures.length; i++) {
            double distance = calculateDistance(trainFeatures[i], feature);
            pq.offer(new Neighbor(distance, trainLabels[i]));
            if (pq.size() > k) {
                pq.poll();
            }
        }

        double[] aggregatedLabel = new double[trainLabels[0].length];
        double totalWeight = 0;
        for (Neighbor neighbor : pq) {
            double weight = 1 / (neighbor.distance + EPSILON);
            for (int i = 0; i < aggregatedLabel.length; i++) {
                aggregatedLabel[i] += neighbor.label[i] * weight;
            }
            totalWeight += weight;
        }

        for (int i = 0; i < aggregatedLabel.length; i++) {
            aggregatedLabel[i] /= totalWeight;
        }
        return aggregatedLabel;
    }

    private double calculateDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    private static class Neighbor {
        private double distance;
        private double[] label;

        public Neighbor(double distance, double[] label) {
            this.distance = distance;
            this.label = label;
        }
    }
}
