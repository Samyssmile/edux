package de.edux.core.math.matrix.strassen;

import static org.junit.jupiter.api.Assertions.*;

import de.edux.core.math.IMatrixArithmetic;
import org.junit.jupiter.api.Test;

class StrassenParallelTest {

  @Test
  public void shouldMultiplyWithParallelStrassen() {
    int matrixSize = 2000;
    double[][] matrixA = new double[matrixSize][matrixSize];
    double[][] matrixB = new double[matrixSize][matrixSize];

    for (int i = 0; i < matrixSize; i++) {
      for (int j = 0; j < matrixSize; j++) {
        matrixA[i][j] = 1;
        matrixB[i][j] = 1;
      }
    }

    IMatrixArithmetic strassenParallel = new StrassenParallel();

    double[][] result = strassenParallel.multiply(matrixA, matrixB);

    for (int i = 0; i < matrixSize; i++) {
      for (int j = 0; j < matrixSize; j++) {
        assertEquals(matrixSize, result[i][j], "Result on [" + i + "][" + j + "] not correct.");
      }
    }
  }
}
