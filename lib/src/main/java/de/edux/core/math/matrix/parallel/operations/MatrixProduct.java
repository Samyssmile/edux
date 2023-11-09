package de.edux.core.math.matrix.parallel.operations;

import de.edux.core.math.IMatrixProduct;
import java.util.stream.IntStream;

public class MatrixProduct implements IMatrixProduct {

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {
    int aRows = matrixA.length;
    int aColumns = matrixA[0].length;
    int bColumns = matrixB[0].length;

    double[][] result = new double[aRows][bColumns];
    IntStream.range(0, aRows)
        .parallel()
        .forEach(
            row -> {
              for (int col = 0; col < bColumns; col++) {
                for (int i = 0; i < aColumns; i++) {
                  result[row][col] += matrixA[row][i] * matrixB[i][col];
                }
              }
            });

    return result;
  }
}
