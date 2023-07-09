package de.edux.ml.knn;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class KnnClassifierTest {

    private KnnClassifier classifier;
    private List<ILabeledPoint> trainingPoints;
    private List<ILabeledPoint> testPoints;

    @BeforeEach
    public void setup() {
        // Erstellen Sie einige Trainingspunkte
        ILabeledPoint trainingPoint1 = new KnnPoint(new double[]{1.0, 1.2, 1.4}, "Label1");
        ILabeledPoint trainingPoint2 = new KnnPoint(new double[]{3.0, 3.1, 3.2}, "Label2");
        ILabeledPoint trainingPoint3 = new KnnPoint(new double[]{7.0, 7.0, 7.0}, "Label3");

        trainingPoints = Arrays.asList(trainingPoint1, trainingPoint2, trainingPoint3);

        // Erstellen Sie einige Testpunkte
        ILabeledPoint testPoint1 = new KnnPoint(new double[]{1.0, 1.0, 1.5}, "Label1");
        ILabeledPoint testPoint2 = new KnnPoint(new double[]{3.0, 3.5, 3.1}, "Label2");
        ILabeledPoint testPoint3 = new KnnPoint(new double[]{7.0, 7.0, 7.0}, "Label3");

        testPoints = Arrays.asList(testPoint1, testPoint2, testPoint3);

        // Erstellen Sie den Klassifikator
        classifier = new KnnClassifier(1, trainingPoints);

        classifier.evaluate(testPoints);
    }

    @Test
    public void testClassify() {
        // Testen Sie die classify() Methode
        assertEquals("Label1", classifier.classify(testPoints.get(0)));
        assertEquals("Label2", classifier.classify(testPoints.get(1)));
        assertEquals("Label3", classifier.classify(testPoints.get(2)));
    }

    @Test
    public void testEvaluate() {
        // Testen Sie die evaluate() Methode
        assertEquals(100, classifier.evaluate(testPoints), 0.01);
    }
}
