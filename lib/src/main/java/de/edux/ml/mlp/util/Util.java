package de.edux.ml.mlp.util;

import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;

public class Util {

    private static final Random random = new Random();

    public static Matrix generateInputMatrix(int rows, int cols) {
        return new Matrix(rows, cols, i -> random.nextGaussian());
    }

    public static Matrix generateExpectedMatrix(int rows, int cols) {
        Matrix expected = new Matrix(rows, cols, i -> 0);
        for (int col = 0; col < cols; col++) {
            int randowmRow = random.nextInt(rows);
            expected.set(randowmRow, col, 1);
        }

        return expected;
    }

    public static Matrix generateTrainableExpectedMatrix(int outputRows, Matrix input) {
        Matrix expected = new Matrix(outputRows, input.getCols());

        Matrix columnSum = input.sumColumns();
        columnSum.forEach((row, col, value) -> {
            int rowIndex = (int) (outputRows * (Math.sin(value) + 1) / 2.0);
            expected.set(rowIndex, col, 1);
        });

        return expected;

    }

    /**
     * Generates a matrix with a gaussian distribution and a radius between 0 and outputRows
     *
     * @param inputRows  number of rows
     * @param outputRows number of rows
     * @param cols       number of columns
     * @return a matrix with a gaussian distribution and a radius between 0 and outputRows
     */
    public static TrainingMatrices generateTrainingMatrices(int inputRows, int outputRows, int cols) {
        var io = generateTrainingArrays(inputRows, outputRows, cols);
        Matrix input = new Matrix(inputRows, cols, io.getInput());
        Matrix output = new Matrix(outputRows, cols, io.getOutput());

        return new TrainingMatrices(input, output);
    }

    public static TrainingArrays generateTrainingArrays(int inputSize, int outputSize, int numberItems) {
        double[] input = new double[inputSize * numberItems];
        double[] output = new double[outputSize * numberItems];

        int inputPos = 0;
        int outputPos = 0;
        for (int col = 0; col < numberItems; col++) {
            int radius = random.nextInt(outputSize);

            double[] values = new double[inputSize];
            double initialRadius = 0;
            for (int row = 0; row < inputSize; row++) {
                values[row] = random.nextGaussian();
                initialRadius += values[row] * values[row];
            }
            initialRadius = Math.sqrt(initialRadius);

            for (int row = 0; row < inputSize; row++) {
                input[inputPos++] = values[row] * radius / initialRadius;

            }
            output[outputPos + radius] = 1;
            outputPos += outputSize;

        }
        return new TrainingArrays(input, output);
    }

}
