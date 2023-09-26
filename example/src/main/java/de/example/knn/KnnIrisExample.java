package de.example.knn;

import de.edux.ml.knn.ILabeledPoint;
import de.edux.ml.knn.KnnClassifier;
import de.edux.ml.knn.KnnPoint;
import de.example.data.iris.Iris;
import de.example.data.iris.IrisProvider;

import java.util.ArrayList;
import java.util.List;

/**
 * Knn - K nearest neighbors
 * Dataset: Iris
 * First transfer the iris data into KnnPoints, use the variety as label. Then use the KnnClassifier to classify the test data.
 */
public class KnnIrisExample {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;

    public static void main(String[] args) {
        var datasetProvider = new IrisProvider(NORMALIZE, SHUFFLE, 0.6);
        datasetProvider.printStatistics();

        List<ILabeledPoint> labeledPoints = new ArrayList<>();
        for (int i = 0; i < datasetProvider.getTrainFeatures().length; i++) {
            labeledPoints.add(new KnnPoint(datasetProvider.getTrainFeatures()[i], datasetProvider.getTrainData().get(i).variety));
        }

        KnnClassifier knnClassifier = new KnnClassifier(1, labeledPoints);

        // Evaluate on test data
        // transfer Iris to KnnPoint
        List<Iris> testDataset = datasetProvider.getTestData();
        List<ILabeledPoint> testLabeledPoints = new ArrayList<>();
        testDataset.forEach(iris -> {
            ILabeledPoint labeledPoint = new KnnPoint(iris.getFeatures(), iris.variety);
            testLabeledPoints.add(labeledPoint);
        });

        //Evaluate
        knnClassifier.evaluate(testLabeledPoints);
    }
}