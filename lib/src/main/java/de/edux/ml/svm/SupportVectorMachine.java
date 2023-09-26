package de.edux.ml.svm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

public class SupportVectorMachine implements ISupportVectorMachine {

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
    public void train(double[][] features, int[] labels) {
        // Identify unique class labels
        Set<Integer> uniqueLabels = Arrays.stream(labels).boxed().collect(Collectors.toSet());
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
                    if (labels[k] == uniqueLabelsArray[i] || labels[k] == uniqueLabelsArray[j]) {
                        list.add(features[k]);
                        // Ensure that the sign of the label matches our assumption
                        pairLabelsList.add(labels[k] == uniqueLabelsArray[i] ? 1 : -1);
                    }
                }
                double[][] pairFeatures = list.toArray(new double[0][]);
                int[] pairLabels = pairLabelsList.stream().mapToInt(Integer::intValue).toArray();

                // Train the model on the pair
                model.train(pairFeatures, pairLabels);
                models.put(key, model);
            }
        }
    }

    @Override
    public int predict(double[] features) {
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
        return voteCount.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }



    @Override
    public double evaluate(double[][] features, int[] labels) {
        int correct = 0;
        for (int i = 0; i < features.length; i++) {
            if (predict(features[i]) == labels[i]) {
                correct++;
            }
        }

        double accuracy = (double) correct / features.length;

        LOG.info("Accuracy: " + String.format("%.4f", accuracy * 100) + "%");
        return accuracy;
    }
}

