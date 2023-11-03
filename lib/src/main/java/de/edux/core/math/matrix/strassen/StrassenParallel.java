package de.edux.core.math.matrix.strassen;

import de.edux.core.math.IMatrixArithmetic;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class StrassenParallel implements IMatrixArithmetic {

  private ForkJoinPool forkJoinPool = new ForkJoinPool(4);

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {
    int n = matrixA.length;
    int m = nextPowerOfTwo(n);
    double[][] extendedMatrixA = new double[m][m];
    double[][] extendedMatrixB = new double[m][m];

    for (int i = 0; i < n; i++) {
      System.arraycopy(matrixA[i], 0, extendedMatrixA[i], 0, matrixA[i].length);
      System.arraycopy(matrixB[i], 0, extendedMatrixB[i], 0, matrixB[i].length);
    }

    double[][] extendedResult =
        forkJoinPool.invoke(new StrassenTask(extendedMatrixA, extendedMatrixB));

    double[][] result = new double[n][n];
    for (int i = 0; i < n; i++) {
      System.arraycopy(extendedResult[i], 0, result[i], 0, n);
    }

    return result;
  }

  private double[][] conventionalMultiply(double[][] A, double[][] B) {
    int n = A.length;
    double[][] result = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return result;
  }

  private int nextPowerOfTwo(int number) {
    int power = 1;
    while (power < number) {
      power *= 2;
    }
    return power;
  }

  private void divideMatrix(double[][] parent, double[][] child, int iB, int jB) {
    for (int i1 = 0, i2 = iB; i1 < child.length; i1++, i2++) {
      for (int j1 = 0, j2 = jB; j1 < child.length; j1++, j2++) {
        child[i1][j1] = parent[i2][j2];
      }
    }
  }

  private void combineMatrix(double[][] child, double[][] parent, int iB, int jB) {
    for (int i1 = 0, i2 = iB; i1 < child.length; i1++, i2++) {
      for (int j1 = 0, j2 = jB; j1 < child.length; j1++, j2++) {
        parent[i2][j2] = child[i1][j1];
      }
    }
  }

  private double[][] addMatrices(double[][] a, double[][] b) {
    int n = a.length;
    double[][] result = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        result[i][j] = a[i][j] + b[i][j];
      }
    }
    return result;
  }

  private double[][] subtractMatrices(double[][] a, double[][] b) {
    int n = a.length;
    double[][] result = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        result[i][j] = a[i][j] - b[i][j];
      }
    }
    return result;
  }

  private class StrassenTask extends RecursiveTask<double[][]> {
    private double[][] A;
    private double[][] B;

    StrassenTask(double[][] A, double[][] B) {
      this.A = A;
      this.B = B;
    }

    @Override
    protected double[][] compute() {
      int n = A.length;

      if (n <= 64) {
        return conventionalMultiply(A, B);
      } else {
        int newSize = n / 2;
        double[][] a11 = new double[newSize][newSize];
        double[][] a12 = new double[newSize][newSize];
        double[][] a21 = new double[newSize][newSize];
        double[][] a22 = new double[newSize][newSize];
        double[][] b11 = new double[newSize][newSize];
        double[][] b12 = new double[newSize][newSize];
        double[][] b21 = new double[newSize][newSize];
        double[][] b22 = new double[newSize][newSize];

        divideMatrix(A, a11, 0, 0);
        divideMatrix(A, a12, 0, newSize);
        divideMatrix(A, a21, newSize, 0);
        divideMatrix(A, a22, newSize, newSize);
        divideMatrix(B, b11, 0, 0);
        divideMatrix(B, b12, 0, newSize);
        divideMatrix(B, b21, newSize, 0);
        divideMatrix(B, b22, newSize, newSize);

        StrassenTask task1 = new StrassenTask(addMatrices(a11, a22), addMatrices(b11, b22));
        StrassenTask task2 = new StrassenTask(addMatrices(a21, a22), b11);
        StrassenTask task3 = new StrassenTask(a11, subtractMatrices(b12, b22));
        StrassenTask task4 = new StrassenTask(a22, subtractMatrices(b21, b11));
        StrassenTask task5 = new StrassenTask(addMatrices(a11, a12), b22);
        StrassenTask task6 = new StrassenTask(subtractMatrices(a21, a11), addMatrices(b11, b12));
        StrassenTask task7 = new StrassenTask(subtractMatrices(a12, a22), addMatrices(b21, b22));

        invokeAll(task1, task2, task3, task4, task5, task6, task7);

        double[][] m1 = task1.join();
        double[][] m2 = task2.join();
        double[][] m3 = task3.join();
        double[][] m4 = task4.join();
        double[][] m5 = task5.join();
        double[][] m6 = task6.join();
        double[][] m7 = task7.join();

        double[][] c11 = addMatrices(subtractMatrices(addMatrices(m1, m4), m5), m7);
        double[][] c12 = addMatrices(m3, m5);
        double[][] c21 = addMatrices(m2, m4);
        double[][] c22 = addMatrices(subtractMatrices(addMatrices(m1, m3), m2), m6);

        double[][] result = new double[n][n];

        combineMatrix(c11, result, 0, 0);
        combineMatrix(c12, result, 0, newSize);
        combineMatrix(c21, result, newSize, 0);
        combineMatrix(c22, result, newSize, newSize);

        return result;
      }
    }
  }
}
