package de.edux.core.math.matrix.strassen;

import static org.junit.jupiter.api.Assertions.*;

import de.edux.core.math.IMatrixArithmetic;
import org.junit.jupiter.api.Test;

class StrassenTest {

  @Test
  public void testLargeMatrixMultiplication() {
    double[][] matrixA = new double[1000][1000];
    double[][] matrixB = new double[1000][1000];

    for (int i = 0; i < 1000; i++) {
      for (int j = 0; j < 1000; j++) {
        matrixA[i][j] = 1;
        matrixB[i][j] = 1;
      }
    }

    IMatrixArithmetic strassen = new Strassen();

    double[][] result = strassen.multiply(matrixA, matrixB);
    for (int i = 0; i < 1000; i++) {
      for (int j = 0; j < 1000; j++) {
        assertEquals(1000, result[i][j], "Result on [" + i + "][" + j + "] not correct.");
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
}
