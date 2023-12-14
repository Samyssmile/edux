package de.edux.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;
import org.junit.jupiter.api.Test;

public class SoftmaxTest {
    final Random random = new Random();

    @Test
    void shouldSoftmax() {

        Matrix a = new Matrix(5, 8, (index) -> random.nextGaussian());
        Matrix result = a.softmax();
        double[] colSums = new double[8];
        result.forEach((row, col, value) -> {
            assertTrue(value >= 0 && value <= 1);
            colSums[col] += value;
        });

        for (int i = 0; i < colSums.length; i++) {
            assertEquals(1, colSums[i], 1e-6);
        }




    }
}
