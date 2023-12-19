package de.edux.core;

import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.ml.mlp.core.network.loss.LossFunctions;
import de.edux.ml.mlp.core.tensor.Matrix;
import org.junit.jupiter.api.Test;

public class CategoricalCrossEntropyLossTest {

    private static final double DELTA = 0.05;

    @Test
    public void shouldCalculateCategoricalCrossEntropyLoss() {
        double[] expectedValues = {1, 0, 0, 0, 0, 1, 0, 1, 0};

        Matrix expected = new Matrix(3, 3, i -> expectedValues[i]);
        Matrix actual = new Matrix(3, 3, i -> DELTA * i * i).softmax();
        Matrix result = LossFunctions.crossEntropy(expected, actual);

        actual.forEach((row, col, index, value) -> {
            double expectedValue = expected.getData()[index];
            double loss = result.getData()[col];

            if (expectedValue > 0.9) {
                assertTrue(Math.abs(-Math.log(value) - loss) < 0.001, String.format("expected: %f, actual: %f", -Math.log(value), loss));
            }
        });


    }
}
