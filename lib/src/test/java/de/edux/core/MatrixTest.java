package de.edux.core;

import static org.junit.jupiter.api.Assertions.*;

import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;
import org.junit.jupiter.api.Test;

class MatrixTest {
    private static final double TOLERANCE = 1e-6;
    private final Random random = new Random();

    @Test
    public void testMultiply() {
        // Creating a matrix with elements initialized to their index - 1
        Matrix a = new Matrix(2, 3, (index) -> index - 1);
        double x = 0.5;

        // Applying the multiplication
        Matrix result = a.apply((index, value) -> value * x);

        // Testing the result
        for (int i = 0; i < 2 * 3; i++) {
            assertEquals((i - 1) * x, result.getData()[i]);
        }
    }

    @Test
    public void testEquals() {
        Matrix a = new Matrix(2, 3, (index) -> index - 6);
        Matrix b = new Matrix(2, 3, (index) -> index - 6);
        Matrix c = new Matrix(2, 3, (index) -> index - 6.2);

        assertEquals(a, b);
        assertNotEquals(a, c);
    }

    @Test
    public void testAddMatrices() {
        Matrix a = new Matrix(2, 3, (index) -> index);
        Matrix b = new Matrix(2, 3, (index) -> index * 2);
        Matrix expected = a.apply((index, value) -> value + b.getData()[index]);

        Matrix result = a.add(b);

        assertEquals(expected, result);
    }

    @Test
    public void shouldNotMultiplyMatricesWithWrongDimensions() {
        Matrix a = new Matrix(2, 2);
        Matrix b = new Matrix(3, 2);
        assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
    }

    @Test
    public void shouldMultiplyMatrices() {
        Matrix a = new Matrix(2, 2, (index) -> index);
        Matrix b = new Matrix(2, 2, (index) -> index * 2);

    /*+-------+-------+
    | 0,000 | 1,000 |
    +-------+-------+
    | 2,000 | 3,000 |
    +-------+-------+
        multiply
    +-------+-------+
    | 0,000 | 2,000 |
    +-------+-------+
    | 4,000 | 6,000 |
    +-------+-------+*/

        Matrix expected = new Matrix(2, 2);
        expected.getData()[0] = 4;
        expected.getData()[1] = 6;
        expected.getData()[2] = 12;
        expected.getData()[3] = 22;

        Matrix result = a.multiply(b);
        assertEquals(expected, result);
    }

    @Test
    public void shouldMultiplyWithDifferentColsAndRows() {
        Matrix a = new Matrix(2, 3, (index) -> index);
        Matrix b = new Matrix(3, 2, (index) -> index);

    /*+-------+-------+-------+
    | 0,000 | 1,000 | 2,000 |
    +-------+-------+-------+
    | 3,000 | 4,000 | 5,000 |
    +-------+-------+-------+
        multiply
    +-------+--------+
    | 0,000 |  2,000 |
    +-------+--------+
    | 4,000 |  6,000 |
    +-------+--------+
    | 8,000 | 10,000 |
    +-------+--------+*/

        Matrix expected = new Matrix(2, 2);
        expected.getData()[0] = 10;
        expected.getData()[1] = 13;
        expected.getData()[2] = 28;
        expected.getData()[3] = 40;

        Matrix result = a.multiply(b);
    /*
    *+--------+--------+
    | 10,000 | 13,000 |
    +--------+--------+
    | 28,000 | 40,000 |
    +--------+--------+
    */
        System.out.println(result);
        assertEquals(expected, result);
    }

    @Test
    public void shouldRunWithoutOutOfMemory() {
        try {
            var matrixSize = 150;
            Matrix a = new Matrix(matrixSize, matrixSize, (index) -> index);
            Matrix b = new Matrix(matrixSize, matrixSize, (index) -> index);

            long startTime = System.nanoTime();
            Matrix result = a.multiply(b);
            long endTime = System.nanoTime();

            System.out.println("Time: " + (endTime - startTime) / 1e9 + "s");
        } catch (OutOfMemoryError e) {
            fail("Test failed due to insufficient memory: " + e.getMessage());
        }
    }

    @Test
    public void shouldRunWithoutOutOfMemoryOnParallelMultiplication() {
        try {
            Matrix a = new Matrix(1500, 1500, (index) -> index);
            Matrix b = new Matrix(1500, 1500, (index) -> index);

            long startTime = System.nanoTime();
            Matrix result = a.multiplyParallel(b);
            long endTime = System.nanoTime();

            System.out.println("Time: " + (endTime - startTime) / 1e9 + "s");
        } catch (OutOfMemoryError e) {
            fail("Test failed due to insufficient memory: " + e.getMessage());
        }
    }

    @Test
    void shouldMultiplyAndAdd() {
        Matrix inout = new Matrix(3, 3, (index) -> index + 1);
        Matrix weights = new Matrix(3, 3, (index) -> index + 1);
        Matrix bias = new Matrix(3, 1, (index) -> index + 1);

        Matrix result = inout.multiply(weights).add(bias);

        Matrix expected = new Matrix(3, 3);

        expected.getData()[0] = 31;
        expected.getData()[1] = 37;
        expected.getData()[2] = 43;
        expected.getData()[3] = 68;
        expected.getData()[4] = 83;
        expected.getData()[5] = 98;
        expected.getData()[6] = 105;
        expected.getData()[7] = 129;
        expected.getData()[8] = 153;

        assert result.equals(expected);
    }

    @Test
    void shouldSumColumns() {
        Matrix a = new Matrix(4, 5, (index) -> index);
        Matrix result = a.sumColumns();
        Matrix expected = new Matrix(1, 5);
        expected.getData()[0] = 30;
        expected.getData()[1] = 34;
        expected.getData()[2] = 38;
        expected.getData()[3] = 42;
        expected.getData()[4] = 46;
        assert result.equals(expected);
    }

    @Test
    void shouldAddIncrement() {
        Matrix a = new Matrix(5, 8, (index) -> random.nextGaussian());
        int row = 3;
        int col = 2;
        double increment = 10;

        Matrix result = a.addIncrement(row, col, increment);

        double incrementedValue = result.get(row, col);
        double originalValue = a.get(row, col);

        assertEquals(a.get(row, col) + increment, result.get(row, col));
        assertTrue(Math.abs(incrementedValue - (originalValue + increment)) < TOLERANCE);
    }

    @Test
    void shouldTranspose() {
        Matrix m = new Matrix(2, 3, i -> i);
        System.out.println(m);
        Matrix result = m.transpose();

        System.out.println(result);
        double[] expectedData = {0, 3, 1, 4, 2, 5};
        Matrix expected = new Matrix(3, 2, i -> expectedData[i]);

        assertEquals(expected, result);
    }

    @Test
    public void shouldCalculateAverageColumn() {
        int rows = 3;
        int cols = 4;
        Matrix m = new Matrix(rows, cols, i -> 2 * i - 3);
        double averageIndex = (cols - 1) / 2.0;

        Matrix expected = new Matrix(rows, 1);
        expected.modify((row, col, value) -> 2 * (row * cols + averageIndex) - 3);

        Matrix result = m.averageColumn();
        assertEquals(expected, result);

    }

    @Test
    void shouldFindGreatestRowNumber() {
        double[] values = {7, -6, -6, 7, 2, 10, 3, -1, 1};
        Matrix m = new Matrix(3,3, i->values[i]);

        Matrix result = m.getGreatestRowNumber();

        double[] expectedValues = {0, 1, 1};
        Matrix expected = new Matrix(3,1, i->expectedValues[i]);

        assertEquals(expected, result);

        System.out.println(m);
        System.out.println(result);
    }
}
