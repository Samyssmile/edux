package de.edux.core.math.matrix.parallel;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MatrixArithmeticTest {

  private static MatrixArithmetic matrixArithmetic;

  @BeforeAll
  static void setUp() {
    matrixArithmetic = new MatrixArithmetic();
  }

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

    double[][] result = matrixArithmetic.multiply(matrixA, matrixB);
    for (int i = 0; i < matrixSize; i++) {
      for (int j = 0; j < matrixSize; j++) {
        assertEquals(matrixSize, result[i][j], "Result on [" + i + "][" + j + "] not correct.");
      }
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

    assertAll(
        () ->
            assertThrows(
                RuntimeException.class, () -> matrixArithmetic.multiply(matrixA, matrixB)),
        () ->
            assertThrows(
                RuntimeException.class, () -> matrixArithmetic.multiply(matrixC, matrixA)));
  }

  @Test
  public void shouldMultiply8x8MatricesCorrectly() {
    double[][] matrixA = {
      {1, 2, 3, 4, 5, 6, 7, 8},
      {8, 7, 6, 5, 4, 3, 2, 1},
      {2, 3, 4, 5, 6, 7, 8, 9},
      {9, 8, 7, 6, 5, 4, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1},
      {2, 2, 2, 2, 2, 2, 2, 2},
      {1, 3, 5, 7, 9, 11, 13, 15},
      {15, 13, 11, 9, 7, 5, 3, 1}
    };
    double[][] matrixB = {
      {1, 0, 0, 0, 1, 0, 0, 0},
      {0, 1, 0, 0, 0, 1, 0, 0},
      {0, 0, 1, 0, 0, 0, 1, 0},
      {0, 0, 0, 1, 0, 0, 0, 1},
      {1, 0, 0, 0, 1, 0, 0, 0},
      {0, 1, 0, 0, 0, 1, 0, 0},
      {0, 0, 1, 0, 0, 0, 1, 0},
      {0, 0, 0, 1, 0, 0, 0, 1}
    };
    double[][] expected = {
      {6, 8, 10, 12, 6, 8, 10, 12},
      {12, 10, 8, 6, 12, 10, 8, 6},
      {8, 10, 12, 14, 8, 10, 12, 14},
      {14, 12, 10, 8, 14, 12, 10, 8},
      {2, 2, 2, 2, 2, 2, 2, 2},
      {4, 4, 4, 4, 4, 4, 4, 4},
      {10, 14, 18, 22, 10, 14, 18, 22},
      {22, 18, 14, 10, 22, 18, 14, 10}
    };

    double[][] result = matrixArithmetic.multiply(matrixA, matrixB);
    assertArrayEquals(
        expected, result, "The 8x8 matrix multiplication did not yield the correct result.");
  }

  @Test
  public void shouldMultiplyWithZeroMatrix() {
    double[][] matrixA = {
      {0, 0},
      {0, 0}
    };
    double[][] matrixB = {
      {1, 2},
      {3, 4}
    };
    double[][] expected = {
      {0, 0},
      {0, 0}
    };

    double[][] result = matrixArithmetic.multiply(matrixA, matrixB);
    assertArrayEquals(expected, result);
  }

  @Test
  public void shouldMultiplyWithIdentityMatrix() {
    double[][] matrixA = {
      {1, 0},
      {0, 1}
    };
    double[][] matrixB = {
      {5, 6},
      {7, 8}
    };

    double[][] result = matrixArithmetic.multiply(matrixA, matrixB);
    assertArrayEquals(matrixB, result);
  }

  @Test
  public void shouldMultiplySmallMatricesCorrectly() {
    double[][] matrixA = {
      {1, 2},
      {3, 4}
    };
    double[][] matrixB = {
      {2, 0},
      {1, 2}
    };
    double[][] expected = {
      {4, 4},
      {10, 8}
    };

    double[][] result = matrixArithmetic.multiply(matrixA, matrixB);
    assertArrayEquals(expected, result);
  }

  @Test
  void shouldSolveMatrixVectorProduct() {
    int matrixSize = 2048;
    double[][] matrixA = new double[matrixSize][matrixSize];
    double[] vector = new double[matrixSize];

    for (int i = 0; i < matrixSize; i++) {
      for (int j = 0; j < matrixSize; j++) {
        matrixA[i][j] = 1;
        vector[i] = 1;
      }
    }

    double[] resultVector = matrixArithmetic.multiply(matrixA, vector);
    for (int i = 0; i < matrixSize; i++) {
      assertEquals(matrixSize, resultVector[i], "Result on [" + i + "][" + i + "] not correct.");
    }
  }

  @Test
  void shouldHandleEmptyMatrix() {
    double[][] matrix = new double[0][0];
    double[] vector = new double[0];
    assertThrows(IllegalArgumentException.class, () -> matrixArithmetic.multiply(matrix, vector));
  }

  @Test
  void shouldHandleMismatchedSizes() {
    double[][] matrix = {{1, 2, 3}, {4, 5, 6}};
    double[] vector = {1, 2};
    assertThrows(IllegalArgumentException.class, () -> matrixArithmetic.multiply(matrix, vector));
  }

  @Test
  void shouldHandleNullMatrix() {
    double[] vector = {1, 2, 3};
    assertThrows(NullPointerException.class, () -> matrixArithmetic.multiply(null, vector));
  }

  @Test
  void shouldMultiplyVectorWithIdentityMatrix() {
    double[][] matrix = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double[] vector = {1, 2, 3};
    double[] expected = {1, 2, 3};
    double[] result = matrixArithmetic.multiply(matrix, vector);
    assertArrayEquals(
      expected, result, "Multiplying with identity matrix should return the original vector.");
  }
}
