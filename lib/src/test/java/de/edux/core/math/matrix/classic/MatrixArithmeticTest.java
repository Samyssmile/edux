package de.edux.core.math.matrix.classic;

import static org.junit.jupiter.api.Assertions.*;

import de.edux.core.math.IMatrixArithmetic;
import org.junit.jupiter.api.Test;

class MatrixArithmeticTest {

  @Test
  public void testLargeMatrixMultiplication() {
    int matrixSize = 200;
    double[][] matrixA = new double[matrixSize][matrixSize];
    double[][] matrixB = new double[matrixSize][matrixSize];

    for (int i = 0; i < matrixSize; i++) {
      for (int j = 0; j < matrixSize; j++) {
        matrixA[i][j] = 1;
        matrixB[i][j] = 1;
      }
    }

    IMatrixArithmetic classic = new MatrixArithmetic();

    double[][] result = classic.multiply(matrixA, matrixB);
    for (int i = 0; i < matrixSize; i++) {
      for (int j = 0; j < matrixSize; j++) {
        assertEquals(matrixSize, result[i][j], "Result on [" + i + "][" + j + "] not correct.");
      }
    }
  }

  private void printMatrix(double[][] result) {
    for (int i = 0; i < result.length; i++) {
      for (int j = 0; j < result.length; j++) {
        System.out.print(result[i][j] + " ");
      }
      System.out.println();
    }
  }

  @Test
  public void shouldThrowRuntimeExceptionForMatricesWithIncompatibleSizes() {
    double[][] matrixA = {
      {3, -5, 1},
      {-2, 0, 4},
      {-1, 6, 5},
    };
    double[][] matrixB = {
      {7, 2, 4},
      {0, 1, -5},
    };
    double[][] matrixC = {
      {3, -5},
      {-2, 0},
      {-1, 6},
    };

    IMatrixArithmetic classic = new MatrixArithmetic();

    assertAll(
        () -> assertThrows(RuntimeException.class, () -> classic.multiply(matrixA, matrixB)),
        () -> assertThrows(RuntimeException.class, () -> classic.multiply(matrixC, matrixA)));
  }
}
