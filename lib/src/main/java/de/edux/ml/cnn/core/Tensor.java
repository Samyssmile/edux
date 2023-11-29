package de.edux.ml.cnn.core;

import java.util.function.Function;

public class Tensor {
  private double[][] data;

  public Tensor(int rows, int cols) {
    data = new double[rows][cols];
  }

  public double[][] getData() {
    return data;
  }

  // Angenommen, Softmax wird auf eine Zeile des Tensors angewendet
  public double[] softmax(int rowIndex) {
    double[] row = data[rowIndex];
    double sum = 0.0;
    double[] exps = new double[row.length];
    for (int i = 0; i < row.length; i++) {
      exps[i] = Math.exp(row[i]);
      sum += exps[i];
    }
    for (int i = 0; i < row.length; i++) {
      exps[i] /= sum;
    }
    return exps;
  }

  // Optional: Softmax auf eine Spalte anwenden
  public double[] softmaxColumn(int colIndex) {
    double sum = 0.0;
    double[] exps = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      exps[i] = Math.exp(data[i][colIndex]);
      sum += exps[i];
    }
    for (int i = 0; i < exps.length; i++) {
      exps[i] /= sum;
    }
    return exps;
  }
  public Tensor getBatch(int startRow, int endRow) {
    // Validierung der Indizes
    if (startRow < 0 || endRow > data.length || startRow >= endRow) {
      throw new IllegalArgumentException("Ungültige Start- oder Endindizes für den Batch");
    }

    int numRows = endRow - startRow;
    int numCols = data[0].length;
    Tensor batch = new Tensor(numRows, numCols);

    for (int i = 0; i < numRows; i++) {
      System.arraycopy(data[startRow + i], 0, batch.getData()[i], 0, numCols);
    }

    return batch;
  }

  // Gibt die Anzahl der Zeilen zurück
  public int getRows() {
    return data.length;
  }

  // Gibt die Anzahl der Spalten zurück
  public int getCols() {
    if (data.length > 0) {
      return data[0].length;
    }
    return 0;
  }

  public Tensor subtract(Tensor other) {
    if (this.data.length != other.data.length || this.data[0].length != other.data[0].length) {
      throw new IllegalArgumentException("Unpassende Matrixgrößen");
    }

    int rows = this.data.length;
    int cols = this.data[0].length;
    Tensor result = new Tensor(rows, cols);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = this.data[i][j] - other.data[i][j];
      }
    }

    return result;
  }
  public Tensor multiply(double learningRate) {
    int rows = data.length;
    int cols = data[0].length;
    Tensor result = new Tensor(rows, cols);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = data[i][j] * learningRate;
      }
    }

    return result;
  }


  public Tensor sumRows() {
    int rows = data.length;
    Tensor result = new Tensor(rows, 1);

    for (int i = 0; i < rows; i++) {
      double sum = 0;
      for (int j = 0; j < data[i].length; j++) {
        sum += data[i][j];
      }
      result.data[i][0] = sum;
    }

    return result;
  }

  public Tensor transpose() {
    int rows = data.length;
    int cols = data[0].length;
    Tensor transposed = new Tensor(cols, rows);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        transposed.data[j][i] = data[i][j];
      }
    }

    return transposed;
  }

  public Tensor deepCopy() {
    Tensor result = new Tensor(data.length, data[0].length);
    for (int i = 0; i < data.length; i++) {
      System.arraycopy(data[i], 0, result.data[i], 0, data[i].length);
    }
    return result;
  }

  public static Tensor dot(Tensor a, Tensor b) {
    if (a.data[0].length != b.data.length) {
      throw new IllegalArgumentException("Unpassende Matrixgrößen");
    }

    Tensor result = new Tensor(a.data.length, b.data[0].length);
    for (int i = 0; i < result.data.length; i++) {
      for (int j = 0; j < result.data[0].length; j++) {
        for (int k = 0; k < a.data[0].length; k++) {
          result.data[i][j] += a.data[i][k] * b.data[k][j];
        }
      }
    }
    return result;
  }

  public Tensor applyFunction(Function<Double, Double> function) {
    Tensor result = new Tensor(this.data.length, this.data[0].length);
    for (int i = 0; i < this.data.length; i++) {
      for (int j = 0; j < this.data[i].length; j++) {
        result.data[i][j] = function.apply(this.data[i][j]);
      }
    }
    return result;
  }

  public Tensor add(Tensor tensor) {
    if (this.data.length != tensor.data.length || this.data[0].length != tensor.data[0].length) {
      throw new IllegalArgumentException("Unpassende Matrixgrößen");
    }
    Tensor result = new Tensor(this.data.length, this.data[0].length);
    for (int i = 0; i < this.data.length; i++) {
      for (int j = 0; j < this.data[i].length; j++) {
        result.data[i][j] = this.data[i][j] + tensor.data[i][j];
      }
    }
    return result;
  }




  public void set(int i, int j, double v) {
    data[i][j] = v;
  }
}
