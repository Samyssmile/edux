package de.example.randomforest;

import de.edux.api.Classifier;
import de.edux.ml.randomforest.RandomForest;
import de.example.data.iris.IrisProvider;

public class RandomForestExample {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;

    public static void main(String[] args) {

        var datasetProvider = new IrisProvider(NORMALIZE, SHUFFLE, 0.6);
        datasetProvider.printStatistics();

        double[][] trainFeatures = datasetProvider.getTrainFeatures();
        double[][] trainLabels = datasetProvider.getTrainLabels();

        Classifier randomForest = new RandomForest(100, 10, 2, 1, 3, 60);
        randomForest.train(trainFeatures, trainLabels);

        double[][] testFeatures = datasetProvider.getTestFeatures();
        double[][] testLabels = datasetProvider.getTestLabels();

        randomForest.evaluate(testFeatures, testLabels);

    }
}
