package de.edux.core.network.optimizer;

import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.ml.mlp.core.network.loss.LossFunctions;
import de.edux.ml.mlp.core.network.optimizer.Approximator;
import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;
import org.junit.jupiter.api.Test;

class ApproximatorTest {

    private final Random random = new Random();

    @Test
    void shouldApproximate() {
        final int rows = 4;
        final int cols = 5;
        Matrix input = new Matrix(rows, cols, i -> random.nextGaussian()).softmax();

        Matrix expected = new Matrix(rows, cols, i -> 0);
        for (int col = 0; col < cols; col++) {
            int randowmRow = random.nextInt(rows);
            expected.set(randowmRow, col, 1);
        }


        Matrix result = Approximator.gradient(input, in -> LossFunctions.crossEntropy(expected, in));

        input.forEach((index, value) -> {
            double resultValue = result.getData()[index];
            double expectedValue = expected.getData()[index];

            if (expectedValue < 0.001) {
                assertTrue(Math.abs(resultValue) < 0.01);
            } else {
                assertTrue(Math.abs(resultValue + 1.0 / value) < 0.01);
            }
        });
    }

    @Test
    void shouldSoftmaxCrossEntropyGradient() {
        final int rows = 4;
        final int cols = 5;

        Matrix input = new Matrix(rows, cols, i -> random.nextGaussian());
        Matrix expected = new Matrix(rows, cols, i -> 0);
        for (int col = 0; col < cols; col++) {
            int randowmRow = random.nextInt(rows);
            expected.set(randowmRow, col, 1);
        }

        Matrix softmaxOutput = input.softmax();
        Matrix result = Approximator.gradient(input, in -> LossFunctions.crossEntropy(expected, in.softmax()));

        result.forEach((index, value) -> {
            double softmaxValue = softmaxOutput.getData()[index];
            double expectedValue = expected.getData()[index];
            assertTrue(Math.abs(value - (softmaxValue - expectedValue)) < 0.01);
        });

    }
}