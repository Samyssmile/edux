package de.edux.core.math.matrix.classic;

import de.edux.core.math.IMatrixArithmetic;
import java.util.Arrays;

public class MatrixArithmetic implements IMatrixArithmetic {
  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {
    int columnsInFirstMatrix = matrixA[0].length;
    int rowsInSecondMatrix = matrixB.length;
    if (columnsInFirstMatrix != rowsInSecondMatrix) {
      throw new RuntimeException(
          "\"The number of columns in the first matrix must be equal to the number of rows in the second matrix.\"");
    }

    int rowsInMatrixC = matrixA.length;
    int columnsInMatrixC = matrixB.length;
    double[][] matrixC = new double[rowsInMatrixC][columnsInMatrixC];
    for (int row = 0; row < rowsInMatrixC; row++) {
      for (int column = 0; column < columnsInMatrixC; column++) {
        matrixC[row][column] =
            vectorDotProduct(getRowVector(matrixA, row), getColumnVector(matrixB, column));
      }
    }
    return matrixC;
  }

  public double[] getRowVector(double[][] matrix, int rowIndex) {
    return matrix[rowIndex];
  }

  public double[] getColumnVector(double[][] matrix, int columnIndex) {
    return Arrays.stream(matrix).mapToDouble((row) -> row[columnIndex]).toArray();
  }

  public double vectorDotProduct(double[] rowVector, double[] columVector) {
    int vectorSize = rowVector.length;
    double vectorDotProduct = 0;
    for (int i = 0; i < vectorSize; i++) {
      vectorDotProduct += rowVector[i] * columVector[i];
    }
    return vectorDotProduct;
  }
}
