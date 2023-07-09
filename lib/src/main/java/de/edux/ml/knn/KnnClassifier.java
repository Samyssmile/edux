package de.edux.ml.knn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

public class KnnClassifier {
    private static final Logger LOG = LoggerFactory.getLogger(KnnClassifier.class);
    private int k;
    private List<ILabeledPoint> trainingPoints;

    public KnnClassifier(int k, List<ILabeledPoint> trainingPoints) {
        this.k = k;
        this.trainingPoints = trainingPoints;
    }

    private double distance(ILabeledPoint a, ILabeledPoint b) {
        double[] aFeatures = a.getFeatures();
        double[] bFeatures = b.getFeatures();

        if (aFeatures.length != bFeatures.length) {
            throw new IllegalArgumentException("Both points must have the same number of features");
        }

        double sum = 0.0;
        for (int i = 0; i < aFeatures.length; i++) {
            sum += Math.pow(aFeatures[i] - bFeatures[i], 2);
        }
        return Math.sqrt(sum);
    }

    public String classify(ILabeledPoint unknown) {
        PriorityQueue<ILabeledPoint> nearestNeighbors = new PriorityQueue<>(
                Comparator.comparingDouble(p -> distance(p, unknown))
        );

        for (ILabeledPoint p : trainingPoints) {
            if (nearestNeighbors.size() < k) {
                nearestNeighbors.add(p);
            } else if (distance(p, unknown) < distance(nearestNeighbors.peek(), unknown)) {
                nearestNeighbors.poll();
                nearestNeighbors.add(p);
            }
        }

        Map<String, Long> labelCounts = nearestNeighbors.stream()
                .collect(Collectors.groupingBy(ILabeledPoint::getLabel, Collectors.counting()));
        return Collections.max(labelCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public double evaluate(List<ILabeledPoint> testPoints) {
        int correct = 0;

        for (ILabeledPoint p : testPoints) {
            if (classify(p).equals(p.getLabel())) {
                correct++;
            }
        }

        double accuracy = (double) correct / testPoints.size()*100;
        LOG.info("Accuracy: {}%", accuracy);
        return accuracy;
    }
}
