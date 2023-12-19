package de.edux.core;

import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;
import org.junit.jupiter.api.Test;

public class ReLuTest {

    private Random random = new Random();

    @Test
    void shouldReLu() {
        final int numberOfNeurons = 50;
        final int numberOfInputs = 60;
        final int inputSize = 50;
        Matrix inout = new Matrix(inputSize, numberOfInputs, (index) -> random.nextDouble());
        Matrix weights = new Matrix(numberOfNeurons, inputSize, (index) -> random.nextGaussian());
        Matrix bias = new Matrix(numberOfNeurons, 1, (index) -> random.nextGaussian());

        Matrix result1 = weights.multiply(inout).add(bias);
        Matrix result2 = weights.multiply(inout).add(bias).relu();

        result2.forEach(
                (index, value) -> {
                    double originalValue = result1.getData()[index];
                    if (originalValue <= 0) {
                        assertTrue(Math.abs(originalValue - value) < 1e-6 || Math.abs(value) < 1e-6);
                    } else {
                        assertTrue(Math.abs(value) < 1e-6 || Math.abs(originalValue - value) < 1e-6);
                    }
                });
    }
}
