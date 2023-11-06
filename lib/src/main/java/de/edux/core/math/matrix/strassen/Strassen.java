package de.edux.core.math.matrix.strassen;

import de.edux.core.math.IMatrixArithmetic;

public class Strassen implements IMatrixArithmetic {

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

    double[][] extendedResult = strassen(extendedMatrixA, extendedMatrixB);

    double[][] result = new double[n][n];
    for (int i = 0; i < n; i++) {
      System.arraycopy(extendedResult[i], 0, result[i], 0, n);
    }

    return result;
  }

  private double[][] strassen(double[][] A, double[][] B) {
    int n = A.length;

    double[][] result = new double[n][n];

    if (n == 1) {
      result[0][0] = A[0][0] * B[0][0];
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

      double[][] m1 = strassen(addMatrices(a11, a22), addMatrices(b11, b22));
      double[][] m2 = strassen(addMatrices(a21, a22), b11);
      double[][] m3 = strassen(a11, subtractMatrices(b12, b22));
      double[][] m4 = strassen(a22, subtractMatrices(b21, b11));
      double[][] m5 = strassen(addMatrices(a11, a12), b22);
      double[][] m6 = strassen(subtractMatrices(a21, a11), addMatrices(b11, b12));
      double[][] m7 = strassen(subtractMatrices(a12, a22), addMatrices(b21, b22));

      double[][] c11 = addMatrices(subtractMatrices(addMatrices(m1, m4), m5), m7);
      double[][] c12 = addMatrices(m3, m5);
      double[][] c21 = addMatrices(m2, m4);
      double[][] c22 = addMatrices(subtractMatrices(addMatrices(m1, m3), m2), m6);

      combineMatrix(c11, result, 0, 0);
      combineMatrix(c12, result, 0, newSize);
      combineMatrix(c21, result, newSize, 0);
      combineMatrix(c22, result, newSize, newSize);
    }

    return result;
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

  private int nextPowerOfTwo(int number) {
    int power = 1;
    while (power < number) {
      power *= 2;
    }
    return power;
  }
}
