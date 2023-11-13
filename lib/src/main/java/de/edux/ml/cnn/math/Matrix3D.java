package de.edux.ml.cnn.math;

import java.util.Random;

public class Matrix3D implements IMatrix3D {
  private static final double ALPHA = 0.01; // Leaky ReLU Faktor

  private double[][][] data;
  private int depth;
  private int rows;
  private int cols;

  public Matrix3D(int depth, int rows, int cols) {
    this.depth = depth;
    this.rows = rows;
    this.cols = cols;
    this.data = new double[depth][rows][cols];
  }

  public static Matrix3D random(int depth, int rows, int cols) {
    Matrix3D matrix = new Matrix3D(depth, rows, cols);
    Random random = new Random();

    for (int d = 0; d < depth; d++) {
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          matrix.set(d, r, c, random.nextDouble()); // Zufällige Werte
        }
      }
    }

    return matrix;
  }

  @Override
  public Matrix3D dot(Matrix3D other) {
    if (this.cols != other.rows || this.depth != other.depth) {
      throw new IllegalArgumentException("Ungültige Matrixdimensionen für die Multiplikation");
    }

    Matrix3D result = new Matrix3D(this.depth, this.rows, other.cols);

    for (int d = 0; d < this.depth; d++) {
      for (int i = 0; i < this.rows; i++) {
        for (int j = 0; j < other.cols; j++) {
          double sum = 0.0;
          for (int k = 0; k < this.cols; k++) {
            sum += this.data[d][i][k] * other.data[d][k][j];
          }
          result.data[d][i][j] = sum;
        }
      }
    }

    return result;
  }

  @Override
  public Matrix3D convolve(Matrix3D kernel, int stride, int padding) {
    // Die Ausgabedimensionen berechnen
    int outputDepth = this.depth;
    int outputRows = (this.rows - kernel.getRows() + 2 * padding) / stride + 1;
    int outputCols = (this.cols - kernel.getCols() + 2 * padding) / stride + 1;

    // Initialisieren der Ausgabematrix
    Matrix3D output = new Matrix3D(outputDepth, outputRows, outputCols);

    for (int d = 0; d < outputDepth; d++) {
      for (int row = 0; row < outputRows; row++) {
        for (int col = 0; col < outputCols; col++) {
          double sum = 0.0;

          for (int kRow = 0; kRow < kernel.getRows(); kRow++) {
            for (int kCol = 0; kCol < kernel.getCols(); kCol++) {
              int rowIdx = row * stride + kRow - padding;
              int colIdx = col * stride + kCol - padding;

              // Prüfen, ob der Index innerhalb der Grenzen der Eingabematrix liegt
              if (rowIdx >= 0 && rowIdx < this.rows && colIdx >= 0 && colIdx < this.cols) {
                sum += this.data[d][rowIdx][colIdx] * kernel.get(d, kRow, kCol);
              }
            }
          }

          output.set(d, row, col, sum);
        }
      }
    }

    return output;
  }

  @Override
  public Matrix3D maxPooling(int poolSize, int stride) {
    // Die Ausgabedimensionen berechnen
    int outputDepth = this.depth;
    int outputRows = (this.rows - poolSize) / stride + 1;
    int outputCols = (this.cols - poolSize) / stride + 1;

    // Initialisieren der Ausgabematrix
    Matrix3D output = new Matrix3D(outputDepth, outputRows, outputCols);

    for (int d = 0; d < outputDepth; d++) {
      for (int row = 0; row < outputRows; row++) {
        for (int col = 0; col < outputCols; col++) {
          double max = Double.NEGATIVE_INFINITY;

          for (int pRow = 0; pRow < poolSize; pRow++) {
            for (int pCol = 0; pCol < poolSize; pCol++) {
              int rowIdx = row * stride + pRow;
              int colIdx = col * stride + pCol;

              // Prüfen, ob der Index innerhalb der Grenzen der Eingabematrix liegt
              if (rowIdx < this.rows && colIdx < this.cols) {
                max = Math.max(max, this.data[d][rowIdx][colIdx]);
              }
            }
          }

          output.set(d, row, col, max);
        }
      }
    }

    return output;
  }

  @Override
  public Matrix3D applyReLU() {
    Matrix3D result = new Matrix3D(this.depth, this.rows, this.cols);

    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          result.data[d][r][c] = Math.max(0, this.data[d][r][c]);
        }
      }
    }

    return result;
  }

  @Override
  public Matrix3D applyReLUBackward(Matrix3D outputGradient) {
    Matrix3D inputGradient = new Matrix3D(this.depth, this.rows, this.cols);

    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          // Der Gradient ist 1 für Werte > 0, sonst 0
          double gradient = this.data[d][r][c] > 0 ? outputGradient.get(d, r, c) : 0.0;
          inputGradient.set(d, r, c, gradient);
        }
      }
    }

    return inputGradient;
  }

  @Override
  public Matrix3D applyLeakyReLU() {
    Matrix3D result = new Matrix3D(this.depth, this.rows, this.cols);

    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          double value = this.data[d][r][c];
          result.data[d][r][c] = value > 0 ? value : ALPHA * value;
        }
      }
    }

    return result;
  }

  @Override
  public Matrix3D flatten() {
    int totalSize = this.depth * this.rows * this.cols;
    Matrix3D flattened = new Matrix3D(1, 1, totalSize);

    int index = 0;
    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          flattened.set(0, 0, index++, this.get(d, r, c));
        }
      }
    }

    return flattened;
  }

  @Override
  public Matrix3D reshapeBack(int originalDepth, int originalRows, int originalCols) {
    if (this.depth * this.rows * this.cols != originalDepth * originalRows * originalCols) {
      throw new IllegalArgumentException("Die Gesamtgröße der Matrix muss gleich bleiben");
    }

    Matrix3D reshaped = new Matrix3D(originalDepth, originalRows, originalCols);
    int index = 0;
    for (int d = 0; d < originalDepth; d++) {
      for (int r = 0; r < originalRows; r++) {
        for (int c = 0; c < originalCols; c++) {
          reshaped.set(d, r, c, this.get(0, 0, index++));
        }
      }
    }

    return reshaped;
  }

  @Override
  public Matrix3D convolveBackprop(Matrix3D gradient, int stride, int padding) {
    // Die Dimensionen der Ausgabematrix berechnen
    int outputDepth = this.depth;
    int outputRows = (gradient.getRows() - 1) * stride + gradient.getRows();
    int outputCols = (gradient.getCols() - 1) * stride + gradient.getCols();

    // Initialisieren der Ausgabematrix
    Matrix3D output = new Matrix3D(outputDepth, outputRows, outputCols);

    for (int d = 0; d < outputDepth; d++) {
      for (int row = 0; row < outputRows; row++) {
        for (int col = 0; col < outputCols; col++) {
          double sum = 0.0;

          for (int gRow = 0; gRow < gradient.getRows(); gRow++) {
            for (int gCol = 0; gCol < gradient.getCols(); gCol++) {
              int rowIdx = row - gRow * stride;
              int colIdx = col - gCol * stride;

              // Prüfen, ob der Index innerhalb der Grenzen der Gradientenmatrix liegt
              if (rowIdx >= 0
                  && rowIdx < gradient.getRows()
                  && colIdx >= 0
                  && colIdx < gradient.getCols()) {
                sum += gradient.get(d, gRow, gCol);
              }
            }
          }

          output.set(d, row, col, sum);
        }
      }
    }

    // Anwendung des Padding, um die richtige Ausgabegröße zu erhalten
    return output.applyPadding(padding);
  }

  @Override
  public Matrix3D applyPadding(int padding) {
    if (padding == 0) {
      return this;
    }

    int paddedRows = this.rows + 2 * padding;
    int paddedCols = this.cols + 2 * padding;
    Matrix3D paddedMatrix = new Matrix3D(this.depth, paddedRows, paddedCols);

    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < paddedRows; r++) {
        for (int c = 0; c < paddedCols; c++) {
          if (r >= padding
              && r < paddedRows - padding
              && c >= padding
              && c < paddedCols - padding) {
            paddedMatrix.set(d, r, c, this.data[d][r - padding][c - padding]);
          }
        }
      }
    }

    return paddedMatrix;
  }

  @Override
  public Matrix3D add(Matrix3D other) {
    if (this.depth != other.depth || this.rows != other.rows || this.cols != other.cols) {
      throw new IllegalArgumentException("Matrix dimensions must match for addition.");
    }

    Matrix3D result = new Matrix3D(this.depth, this.rows, this.cols);
    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          result.data[d][r][c] = this.data[d][r][c] + other.data[d][r][c];
        }
      }
    }

    return result;
  }

  @Override
  public Matrix3D subtract(Matrix3D other) {

    if (this.depth != other.depth || this.rows != other.rows || this.cols != other.cols) {
      System.out.println("other: " + other);
      System.out.println("this: " + this);
      throw new IllegalArgumentException("Matrix dimensions must match for subtraction.");
    }

    Matrix3D result = new Matrix3D(this.depth, this.rows, this.cols);
    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          result.data[d][r][c] = this.data[d][r][c] - other.data[d][r][c];
        }
      }
    }

    return result;
  }

  @Override
  public Matrix3D multiplyElementWise(Matrix3D other) {
    if (this.depth != other.depth || this.rows != other.rows || this.cols != other.cols) {
      throw new IllegalArgumentException(
          "Matrix dimensions must match for element-wise multiplication.");
    }

    Matrix3D result = new Matrix3D(this.depth, this.rows, this.cols);
    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          result.data[d][r][c] = this.data[d][r][c] * other.data[d][r][c];
        }
      }
    }

    return result;
  }

  @Override
  public void normalize(double mean, double std) {
    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          this.data[d][r][c] = (this.data[d][r][c] - mean) / std;
        }
      }
    }
  }

  @Override
  public Matrix3D transpose() {
    Matrix3D transposed = new Matrix3D(this.depth, this.cols, this.rows);
    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          transposed.data[d][c][r] = this.data[d][r][c];
        }
      }
    }
    return transposed;
  }

  @Override
  public Matrix3D multiply(double value) {
    Matrix3D result = new Matrix3D(this.depth, this.rows, this.cols);
    for (int d = 0; d < this.depth; d++) {
      for (int r = 0; r < this.rows; r++) {
        for (int c = 0; c < this.cols; c++) {
          result.data[d][r][c] = this.data[d][r][c] * value;
        }
      }
    }
    return result;
  }

  @Override
  public Matrix3D sumColumns() {
    Matrix3D columnSums = new Matrix3D(this.depth, 1, this.cols);
    for (int d = 0; d < this.depth; d++) {
      for (int c = 0; c < this.cols; c++) {
        double sum = 0.0;
        for (int r = 0; r < this.rows; r++) {
          sum += this.data[d][r][c];
        }
        columnSums.data[d][0][c] = sum;
      }
    }
    return columnSums;
  }

  @Override
  public double get(int depth, int row, int col) {
    return this.data[depth][row][col];
  }

  @Override
  public void set(int depth, int row, int col, double value) {
    this.data[depth][row][col] = value;
  }

  @Override
  public int getDepth() {
    return depth;
  }

  public void setDepth(int i) {
    this.depth = i;
  }

  @Override
  public int getRows() {
    return rows;
  }

  @Override
  public int getCols() {
    return cols;
  }

  public double[][][] getData() {
    return data;
  }

  @Override
  public Matrix3D sum(int axis) {
    switch (axis) {
      case 0:
        return sumOverDepth();
      case 1:
        return sumOverRows();
      case 2:
        return sumOverCols();
      default:
        throw new IllegalArgumentException("Ungültiger Achsenwert: " + axis);
    }
  }

  @Override
  public Matrix3D sumOverDepth() {
    Matrix3D result = new Matrix3D(1, this.rows, this.cols);
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        double sum = 0;
        for (int k = 0; k < this.depth; k++) {
          sum += this.data[k][i][j];
        }
        result.data[0][i][j] = sum;
      }
    }
    return result;
  }

  @Override
  public Matrix3D sumOverRows() {
    Matrix3D result = new Matrix3D(this.depth, 1, this.cols);
    for (int k = 0; k < this.depth; k++) {
      for (int j = 0; j < this.cols; j++) {
        double sum = 0;
        for (int i = 0; i < this.rows; i++) {
          sum += this.data[k][i][j];
        }
        result.data[k][0][j] = sum;
      }
    }
    return result;
  }

  @Override
  public Matrix3D sumOverCols() {
    Matrix3D result = new Matrix3D(this.depth, this.rows, 1);
    for (int k = 0; k < this.depth; k++) {
      for (int i = 0; i < this.rows; i++) {
        double sum = 0;
        for (int j = 0; j < this.cols; j++) {
          sum += this.data[k][i][j];
        }
        result.data[k][i][0] = sum;
      }
    }
    return result;
  }

  @Override
  public String toString() {
    return "Matrix3D{depth=" + depth + ", rows=" + rows + ", cols=" + cols + '}';
  }
}
