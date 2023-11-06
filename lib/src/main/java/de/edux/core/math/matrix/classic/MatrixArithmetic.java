package de.edux.core.math.matrix.classic;

import de.edux.core.math.IMatrixArithmetic;
import java.util.stream.IntStream;

public class MatrixArithmetic implements IMatrixArithmetic {

  private void checkSizeForMultiplication(double[][] matrixA, double[][] matrixB) {
    int m = matrixB.length;
    int n = matrixA[0].length;
    if (m != n) {
      throw new RuntimeException(
          "\"The number of columns in the first matrix must be equal to the number of rows in the second matrix.\"");
    }
  }

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {
    checkSizeForMultiplication(matrixA, matrixB);

    int aRows = matrixA.length;
    int aColumns = matrixA[0].length;
    int bColumns = matrixB[0].length;

    double[][] result = new double[aRows][bColumns];

    IntStream.range(0, aRows)
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
