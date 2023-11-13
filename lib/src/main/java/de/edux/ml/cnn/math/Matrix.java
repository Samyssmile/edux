package de.edux.ml.cnn.math;

import java.util.Random;

public class Matrix {
  private final double[][] data;
  private final int rows;
  private final int cols;

  public Matrix(double[][] data) {
    this.data = data;
    this.rows = data.length;
    this.cols = data[0].length;
  }

  public static Matrix random(int outputSize, int inputSize) {
    Random random = new Random();
    double[][] data = new double[outputSize][inputSize];
    for (int i = 0; i < outputSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        data[i][j] = random.nextDouble(); // Zufällige Werte zwischen 0.0 und 1.0
      }
    }
    return new Matrix(data);
  }

  public static Matrix ones(int numFeatures, int i) {
    double[][] data = new double[numFeatures][i];
    for (int row = 0; row < numFeatures; row++) {
      for (int col = 0; col < i; col++) {
        data[row][col] = 1.0;
      }
    }
    return new Matrix(data);
  }

  public static Matrix randomBinary(int rows, int cols, double probability) {
    Random random = new Random();
    double[][] data = new double[rows][cols];
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        data[row][col] = random.nextDouble() < probability ? 1.0 : 0.0;
      }
    }
    return new Matrix(data);
  }
  public Matrix dot(Matrix other) {
    if (this.cols != other.rows) {
      throw new IllegalArgumentException(
          "Die Anzahl der Spalten der ersten Matrix muss gleich der Anzahl der Zeilen der zweiten Matrix sein");
    }

    double[][] result = new double[this.rows][other.cols];
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < other.cols; j++) {
        double sum = 0.0;
        for (int k = 0; k < this.cols; k++) {
          sum += this.data[i][k] * other.data[k][j];
        }
        result[i][j] = sum;
      }
    }
    return new Matrix(result);
  }

  public Matrix add(Matrix other) {
    if (this.cols != other.cols && other.rows != 1) {
      throw new IllegalArgumentException("Matrix inner dimensions must agree for addition.");
    }

    double[][] result = new double[this.rows][this.cols];
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        double otherValue = other.rows == 1 ? other.data[0][j] : other.data[i][j];
        result[i][j] = this.data[i][j] + otherValue;
      }
    }
    return new Matrix(result);
  }

  public Matrix multiplyElementWise(Matrix other) {
    double[][] result = new double[this.rows][this.cols];
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        result[i][j] = this.data[i][j] * other.data[i][j];
      }
    }
    return new Matrix(result);
  }

  public Matrix multiply(double v) {
    double[][] result = new double[this.rows][this.cols];
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        result[i][j] = this.data[i][j] * v;
      }
    }
    return new Matrix(result);
  }

  public Matrix transpose() {
    double[][] result = new double[this.cols][this.rows];
    for (int i = 0; i < this.cols; i++) {
      for (int j = 0; j < this.rows; j++) {
        result[i][j] = this.data[j][i];
      }
    }
    return new Matrix(result);
  }

  public Matrix reshape(int newRows, int newCols) {
    if (this.rows * this.cols != newRows * newCols) {
      throw new IllegalArgumentException("Ungültige neue Dimensionen für reshape");
    }

    double[][] newData = new double[newRows][newCols];
    int currentRow = 0, currentCol = 0;

    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        newData[currentRow][currentCol] = this.data[i][j];
        currentCol++;
        if (currentCol == newCols) {
          currentCol = 0;
          currentRow++;
        }
      }
    }

    return new Matrix(newData);
  }

  public Matrix flatten() {
    double[][] flattenedData = new double[1][this.rows * this.cols];
    int index = 0;

    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        flattenedData[0][index++] = this.data[i][j];
      }
    }

    return new Matrix(flattenedData);
  }

  public Matrix subtract(Matrix other) {
    if (this.rows != other.rows || this.cols != other.cols) {
      throw new IllegalArgumentException("Die Dimensionen der Matrizen müssen übereinstimmen");
    }

    double[][] result = new double[this.rows][this.cols];
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        result[i][j] = this.data[i][j] - other.data[i][j];
      }
    }

    return new Matrix(result);
  }

  public double[][] getData() {
    return data;
  }

  public int getRows() {
    return rows;
  }

  public int getCols() {
    return cols;
  }
}
