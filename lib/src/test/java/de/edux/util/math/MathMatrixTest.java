package de.edux.util.math;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class MathMatrixTest {
  private static final long someMaximumValue = 1_000_000_000; // Example value
  private static final Logger LOG = LoggerFactory.getLogger(MathMatrixTest.class);

  static void assertArrayEquals(double[][] expected, double[][] actual) {
    assertEquals(expected.length, actual.length);

    for (int i = 0; i < expected.length; i++) {
      assertArrayEquals(expected[i], actual[i]);
    }
  }

  static void assertArrayEquals(double[] expected, double[] actual) {
    assertEquals(expected.length, actual.length);

    for (int i = 0; i < expected.length; i++) {
      assertEquals(expected[i], actual[i]);
    }
  }

  @Test
  void multiplyMatrices() throws IncompatibleDimensionsException {
    long startTime = System.currentTimeMillis();
    int size = 500;

    double[][] matrixA = generateMatrix(size);
    double[][] matrixB = generateMatrix(size);

    ConcurrentMatrixMultiplication matrixMultiplier = new MathMatrix();
    double[][] resultMatrix = matrixMultiplier.multiplyMatrices(matrixA, matrixB);

    assertEquals(size, resultMatrix.length);
    assertEquals(size, resultMatrix[0].length);

    long endTime = System.currentTimeMillis();
    long timeElapsed = endTime - startTime;
    LOG.info("Time elapsed: " + timeElapsed / 1000 + " seconds");
  }

  @Test
  void multiplyMatricesSmall() throws IncompatibleDimensionsException {
    double[][] matrixA = {
      {1, 2},
      {3, 4}
    };

    double[][] matrixB = {
      {2, 0},
      {1, 3}
    };

    ConcurrentMatrixMultiplication matrixMultiplier = new MathMatrix();
    double[][] resultMatrix = matrixMultiplier.multiplyMatrices(matrixA, matrixB);

    double[][] expectedMatrix = {
      {4, 6},
      {10, 12}
    };

    assertArrayEquals(expectedMatrix, resultMatrix);
  }

  double[][] generateMatrix(int size) {
    double[][] matrix = new double[size][size];
    final int MAX_THREADS = 32;

    ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    List<Future<Void>> futures = new ArrayList<>();

    try {
      int rowsPerThread = Math.max(size / MAX_THREADS, 1);

      for (int i = 0; i < MAX_THREADS && i * rowsPerThread < size; i++) {
        final int startRow = i * rowsPerThread;
        final int endRow = Math.min((i + 1) * rowsPerThread, size);

        futures.add(
            executor.submit(
                () -> {
                  for (int row = startRow; row < endRow; row++) {
                    for (int col = 0; col < size; col++) {
                      matrix[row][col] = Math.random() * 10; // Random values between 0 and 10
                    }
                  }
                  return null;
                }));
      }

      for (Future<Void> future : futures) {
        future.get();
      }
    } catch (InterruptedException | ExecutionException e) {
      e.printStackTrace();
    } finally {
      executor.shutdown();
    }

    return matrix;
  }
}
