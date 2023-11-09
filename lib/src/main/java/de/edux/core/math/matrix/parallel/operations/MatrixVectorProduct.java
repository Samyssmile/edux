package de.edux.core.math.matrix.parallel.operations;

import de.edux.core.math.IMatrixVectorProduct;
import java.util.stream.IntStream;

public class MatrixVectorProduct implements IMatrixVectorProduct {

  @Override
  public double[] multiply(double[][] matrix, double[] vector) {
    int matrixRows = matrix.length;
    int matrixColumns = matrix[0].length;

    double[] result = new double[matrixRows];
    IntStream.range(0, matrixRows)
        .parallel()
        .forEach(
            row -> {
              for (int col = 0; col < matrixColumns; col++) {
                result[row] += matrix[row][col] * vector[col];
              }
            });

    return result;
  }
}
