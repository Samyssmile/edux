package de.nexent.edux.functions.loss;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LossFunctionTest {

    private static final double DELTA = 1e-6; // used to compare floating point numbers

    @Test
    public void testCategoricalCrossEntropy() {
        double[] output = {0.1, 0.2, 0.7};
        double[] target = {0, 0, 1};

        double expectedError = -Math.log(0.7);
        assertEquals(expectedError, LossFunction.CATEGORICAL_CROSS_ENTROPY.calculateError(output, target), DELTA);
    }

    @Test
    public void testMeanSquaredError() {
        double[] output = {0.1, 0.2, 0.7};
        double[] target = {0.2, 0.1, 0.8};

        double expectedError = (Math.pow(0.1, 2) + Math.pow(0.1, 2) + Math.pow(0.1, 2)) / 3;
        assertEquals(expectedError, LossFunction.MEAN_SQUARED_ERROR.calculateError(output, target), DELTA);
    }

    @Test
    public void testMeanAbsoluteError() {
        double[] output = {0.1, 0.2, 0.7};
        double[] target = {0.2, 0.1, 0.8};

        double expectedError = (Math.abs(0.1) + Math.abs(0.1) + Math.abs(0.1)) / 3;
        assertEquals(expectedError, LossFunction.MEAN_ABSOLUTE_ERROR.calculateError(output, target), DELTA);
    }

    @Test
    public void testHingeLoss() {
        double[] output = {0.1, 0.2, 0.7};
        double[] target = {-1, -1, 1};

        double expectedError = (Math.max(0, 1 - (-1) * 0.1) + Math.max(0, 1 - (-1) * 0.2) + Math.max(0, 1 - 1 * 0.7)) / 3;
        assertEquals(expectedError, LossFunction.HINGE_LOSS.calculateError(output, target), DELTA);
    }

    @Test
    public void testSquaredHingeLoss() {
        double[] output = {0.1, 0.2, 0.7};
        double[] target = {-1, -1, 1};

        double expectedError = (Math.pow(Math.max(0, 1 - (-1) * 0.1), 2) + Math.pow(Math.max(0, 1 - (-1) * 0.2), 2) + Math.pow(Math.max(0, 1 - 1 * 0.7), 2)) / 3;
        assertEquals(expectedError, LossFunction.SQUARED_HINGE_LOSS.calculateError(output, target), DELTA);
    }

    @Test
    public void testBinaryCrossEntropy() {
        double[] output = {0.1, 0.2, 0.7};
        double[] target = {1, 0, 1};

        double expectedError = - (Math.log(0.1) + Math.log(1 - 0.2) + Math.log(0.7));
        assertEquals(expectedError, LossFunction.BINARY_CROSS_ENTROPY.calculateError(output, target), DELTA);
    }
}
