package de.edux.ml.randomforest;

import de.edux.ml.decisiontree.DecisionTree;
import de.edux.ml.decisiontree.IDecisionTree;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

public class RandomForest {
    private static final Logger LOG = LoggerFactory.getLogger(RandomForest.class);

    private final List<IDecisionTree> trees = new ArrayList<>();
    private final ThreadLocalRandom threadLocalRandom = ThreadLocalRandom.current();

    public void train(int numTrees,
                      double[][] features,
                      double[][] labels,
                      int maxDepth,
                      int minSamplesSplit,
                      int minSamplesLeaf,
                      int maxLeafNodes,
                      int numberOfFeatures) {

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        List<Future<IDecisionTree>> futures = new ArrayList<>();

        for (int i = 0; i < numTrees; i++) {
            futures.add(executor.submit(() -> {
                IDecisionTree tree = new DecisionTree();
                Sample subsetSample = getRandomSubset(numberOfFeatures, features, labels);
                tree.train(subsetSample.featureSamples(), subsetSample.labelSamples(), maxDepth, minSamplesSplit, minSamplesLeaf, maxLeafNodes);
                return tree;
            }));
        }

        for (Future<IDecisionTree> future : futures) {
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


    public double predict(double[] feature) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Future<Double>> futures = new ArrayList<>();

        for (IDecisionTree tree : trees) {
            futures.add(executor.submit(() -> tree.predict(feature)));
        }

        Map<Double, Long> voteMap = new HashMap<>();
        for (Future<Double> future : futures) {
            try {
                double prediction = future.get();
                voteMap.merge(prediction, 1L, Long::sum);
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

        return voteMap.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new RuntimeException("Failed to find the most common prediction"));
    }


    public double evaluate(double[][] features, double[][] labels) {
        int correctPredictions = 0;
        for (int i = 0; i < features.length; i++) {
            double predictedLabel = predict(features[i]);
            double actualLabel = getIndexOfHighestValue(labels[i]);
            if (predictedLabel == actualLabel) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / features.length;
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
