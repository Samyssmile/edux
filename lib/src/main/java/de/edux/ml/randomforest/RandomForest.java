package de.edux.ml.randomforest;

import de.edux.api.Classifier;
import de.edux.ml.decisiontree.DecisionTree;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * RandomForest Classifier
 * RandomForest is an ensemble learning method, which constructs a multitude of decision trees
 * at training time and outputs the class that is the mode of the classes output by
 * individual trees, or a mean prediction of the individual trees (regression).
 * <p>
 * <b>Note:</b> Training and prediction are performed in a parallel manner using thread pooling.
 * RandomForest handles the training of individual decision trees and their predictions, and
 * determines the final prediction by voting (classification) or averaging (regression) the
 * outputs of all the decision trees in the forest. RandomForest is particularly well suited
 * for multiclass classification and regression on datasets with complex structures.
 * <p>
 * Usage example:
 * <pre>
 * {@code
 * RandomForest forest = new RandomForest();
 * forest.train(numTrees, features, labels, maxDepth, minSamplesSplit, minSamplesLeaf,
 *              maxLeafNodes, numberOfFeatures);
 * double prediction = forest.predict(sampleFeatures);
 * double accuracy = forest.evaluate(testFeatures, testLabels);
 * }
 * </pre>
 * <p>
 * <b>Thread Safety:</b> This class uses concurrent features but may not be entirely thread-safe
 * and should be used with caution in a multithreaded environment.
 * <p>
 * Use {@link #train(double[][], double[][])} to train the forest,
 * {@link #predict(double[])} to predict a single sample, and {@link #evaluate(double[][], double[][])}
 * to evaluate accuracy against a test set.
 */
public class RandomForest implements Classifier {
    private static final Logger LOG = LoggerFactory.getLogger(RandomForest.class);

    private final List<Classifier> trees = new ArrayList<>();
    private final ThreadLocalRandom threadLocalRandom = ThreadLocalRandom.current();
    private final int numTrees;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final int maxLeafNodes;
    private final int numberOfFeatures;

    public RandomForest(int numTrees, int maxDepth,
                 int minSamplesSplit,
                 int minSamplesLeaf,
                 int maxLeafNodes,
                 int numberOfFeatures) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxLeafNodes = maxLeafNodes;
        this.numberOfFeatures = numberOfFeatures;
    }

    public boolean train(double[][] features, double[][] labels) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        List<Future<Classifier>> futures = new ArrayList<>();

        for (int i = 0; i < numTrees; i++) {
            futures.add(executor.submit(() -> {
                Classifier tree = new DecisionTree(maxDepth, minSamplesSplit, minSamplesLeaf, maxLeafNodes);
                Sample subsetSample = getRandomSubset(numberOfFeatures, features, labels);
                tree.train(subsetSample.featureSamples(), subsetSample.labelSamples());
                return tree;
            }));
        }

        for (Future<Classifier> future : futures) {
            try {
                trees.add(future.get());
            } catch (ExecutionException | InterruptedException e) {
                LOG.error("Failed to train a decision tree. Thread: " +
                        Thread.currentThread().getName(), e);
            }
        }
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException ex) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        return true;
    }

    private Sample getRandomSubset(int numberOfFeatures, double[][] features, double[][] labels) {
        if (numberOfFeatures > features.length) {
            throw new IllegalArgumentException("Number of feature must be between 1 and amount of features");
        }
        double[][] subFeatures = new double[numberOfFeatures][];
        double[][] subLabels = new double[numberOfFeatures][];
        for (int i = 0; i < numberOfFeatures; i++) {
            int randomIndex = threadLocalRandom.nextInt(numberOfFeatures);
            subFeatures[i] = features[randomIndex];
            subLabels[i] = labels[randomIndex];
        }

        return new Sample(subFeatures, subLabels);
    }


    @Override
    public double[] predict(double[] feature) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Future<double[]>> futures = new ArrayList<>();

        for (Classifier tree : trees) {
            futures.add(executor.submit(() -> tree.predict(feature)));
        }

        Map<Double, Long> voteMap = new HashMap<>();
        for (Future<double[]> future : futures) {
            try {
                double[] prediction = future.get();
                double label = getIndexOfHighestValue(prediction);
                voteMap.merge(label, 1L, Long::sum);
            } catch (InterruptedException | ExecutionException e) {
                LOG.error("Failed to retrieve prediction from future task. Thread: " +
                        Thread.currentThread().getName(), e);
            }
        }

        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException ex) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        double predictionLabel = voteMap.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get()
                .getKey();

        double[] prediction = new double[trees.get(0).predict(feature).length];
        prediction[(int) predictionLabel] = 1;
        return prediction;
    }

    @Override
    public double evaluate(double[][] features, double[][] labels) {
        int correctPredictions = 0;
        for (int i = 0; i < features.length; i++) {
            double[] predictedLabelProbabilities = predict(features[i]);
            double predictedLabel = getIndexOfHighestValue(predictedLabelProbabilities);
            double actualLabel = getIndexOfHighestValue(labels[i]);
            if (predictedLabel == actualLabel) {
                correctPredictions++;
            }
        }
        double accuracy = (double) correctPredictions / features.length;
        LOG.info("RandomForest - Accuracy: " + String.format("%.4f", accuracy * 100) + "%");
        return accuracy;
    }


    private double getIndexOfHighestValue(double[] labels) {
        int maxIndex = 0;
        double maxValue = labels[0];

        for (int i = 1; i < labels.length; i++) {
            if (labels[i] > maxValue) {
                maxValue = labels[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}
