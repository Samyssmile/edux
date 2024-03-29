package de.edux.ml.mlp.core.tensor;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Matrix implements Serializable {

  private static final String NUMBER_FORMAT = "%.3f";
  private final int rows;
  private final int cols;
  private double tolerance = 1e-5;
  private double[] data;

  public Matrix(int rows, int cols) {
    data = new double[rows * cols];
    this.rows = rows;
    this.cols = cols;
  }

  public Matrix(int rows, int cols, Producer producer) {
    this(rows, cols);
    for (int i = 0; i < data.length; i++) {
      data[i] = producer.produce(i);
    }
  }

  public Matrix(int rows, int cols, double[] values) {
    this.rows = rows;
    this.cols = cols;

    Matrix temp = new Matrix(cols, rows);
    temp.data = values;
    Matrix transposed = temp.transpose();
    data = transposed.data;
  }

  public Matrix(double[][] values) {
    this.rows = values.length;
    this.cols = values[0].length;
    this.data = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(values[i], 0, this.data, i * cols, cols);
    }
  }

  public Matrix multiplyParallel(double rate) {
    Matrix result = new Matrix(this.rows, this.cols);

    IntStream.range(0, this.rows)
        .parallel()
        .forEach(
            i -> {
              for (int j = 0; j < this.cols; j++) {
                result.data[i * this.cols + j] = this.data[i * this.cols + j] * rate;
              }
            });

    return result;
  }

  public double sum() {
    double sum = 0;
    for (double datum : data) {
      sum += datum;
    }
    return sum;
  }

  public Matrix divide(int batches) { // TODO replace with apply
    if (batches == 0) {
      throw new IllegalArgumentException("Division durch null ist nicht erlaubt.");
    }

    Matrix result = new Matrix(this.rows, this.cols);
    for (int i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i] / batches;
    }
    return result;
  }

  /**
   * Subtracts the given matrix from this matrix.
   *
   * <p>This method performs an element-wise subtraction between two matrices. It requires that both
   * matrices have the same dimensions. If the matrices do not have the same dimensions, an
   * IllegalArgumentException is thrown.
   *
   * @param matrix The matrix to be subtracted from this matrix.
   * @return A new Matrix object representing the result of the subtraction.
   * @throws IllegalArgumentException if the input matrix and this matrix do not have the same
   *     dimensions.
   */
  public Matrix subtract(Matrix matrix) {
    if (this.rows != matrix.rows || this.cols != matrix.cols) {
      throw new IllegalArgumentException("Matrices must have the same size.");
    }

    Matrix result = new Matrix(this.rows, this.cols);
    for (int i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i] - matrix.getData()[i];
    }
    return result;
  }

  public Matrix relu() {
    return this.apply((index, value) -> Math.max(0, value));
  }

  public Matrix reluDerivative(Matrix input) {
    return this.apply((index, value) -> input.get(index) > 0 ? value : 0);
  }

  public void set(int row, int col, double value) {
    data[row * cols + col] = value;
  }

  public double get(int row, int col) {
    return data[row * cols + col];
  }

  public Matrix addIncrement(int row, int col, double increment) {
    Matrix result = apply((index, value) -> data[index]);
    double originalValue = result.get(row, col);
    double newValue = originalValue + increment;
    result.set(row, col, newValue);

    return result;
  }

  public Matrix transpose() {
    Matrix result = new Matrix(cols, rows);
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        result.data[col * rows + row] = data[row * cols + col];
      }
    }
    return result;
  }

  public Matrix transposeParallel() {
    Matrix result = new Matrix(cols, rows);

    IntStream.range(0, rows)
        .parallel()
        .forEach(
            row -> {
              for (int col = 0; col < cols; col++) {
                result.data[col * rows + row] = data[row * cols + col];
              }
            });

    return result;
  }

  public double get(int index) {
    return this.getData()[index];
  }

  public Matrix multiply(double rate) {
    return this.apply((index, value) -> value * rate);
  }

  public double[] getData() {
    return data;
  }

  public Matrix apply(IndexValueProducer function) {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < data.length; i++) {
      result.data[i] = function.produce(i, data[i]);
    }
    return result;
  }

  public Matrix multiply(Matrix other) {
    if (cols != other.rows) {
      throw new IllegalArgumentException("Matrix dimensions do not match");
    }
    Matrix result = new Matrix(rows, other.cols);
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < other.cols; col++) {
        double sum = 0;
        for (int i = 0; i < cols; i++) {
          sum += data[row * cols + i] * other.data[i * other.cols + col];
        }
        result.data[row * other.cols + col] = sum;
      }
    }
    return result;
  }

  public Matrix multiplyParallel(Matrix other) {
    if (cols != other.rows) {
      throw new IllegalArgumentException("Matrix dimensions do not match");
    }
    Matrix result = new Matrix(rows, other.cols);
    IntStream.range(0, rows)
        .parallel()
        .forEach(
            row -> {
              for (int col = 0; col < other.cols; col++) {
                double sum = 0;
                for (int i = 0; i < cols; i++) {
                  sum += data[row * cols + i] * other.data[i * other.cols + col];
                }
                result.data[row * other.cols + col] = sum;
              }
            });

    return result;
  }

  public Matrix averageColumn() {
    Matrix result = new Matrix(rows, 1);
    forEach((row, col, value) -> result.data[row] += value / cols);
    return result;
  }

  public Matrix add(Matrix other) {
    // Überprüfen, ob die andere Matrix eine Spaltenmatrix ist, die als Bias verwendet werden kann
    if (this.cols != other.cols && other.cols != 1) {
      throw new IllegalArgumentException(
          "Für die Addition muss die zweite Matrix entweder dieselbe Größe haben oder eine Spaltenmatrix sein.");
    }

    Matrix result = new Matrix(rows, cols);
    for (int row = 0; row < this.rows; row++) {
      for (int col = 0; col < this.cols; col++) {
        if (other.cols == 1) {
          // Addiere den Bias, wenn die zweite Matrix eine Spaltenmatrix ist
          result.data[row * cols + col] = this.data[row * cols + col] + other.data[row];
        } else {
          // Normale elementweise Addition, wenn die zweite Matrix dieselbe Größe hat
          result.data[row * cols + col] =
              this.data[row * cols + col] + other.data[row * cols + col];
        }
      }
    }

    return result;
  }

  public Matrix modify(RowColumnProducer function) {
    int index = 0;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++, index++) {
        data[index] = function.produce(row, col, data[index]);
      }
    }
    return this;
  }

  public void forEach(IndexValueConsumer consumer) {
    for (int i = 0; i < data.length; i++) {
      consumer.consume(i, data[i]);
    }
  }

  public void forEach(RowColIndexValueConsumer consumer) {
    int index = 0;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        consumer.consume(row, col, index, data[index++]);
      }
    }
  }

  public void forEach(RowColValueConsumer consumer) {
    int index = 0;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        consumer.consume(row, col, data[index++]);
      }
    }
  }

  public void setTolerance(double tolerance) {
    this.tolerance = tolerance;
  }

  public Matrix sumColumns() {
    Matrix result = new Matrix(1, cols);
    int index = 0;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        result.data[col] += data[index++];
      }
    }

    return result;
  }

  public Matrix softmax() {
    Matrix result = new Matrix(rows, cols, i -> Math.exp(data[i]));
    Matrix colSum = result.sumColumns();

    result.modify((row, col, value) -> value / colSum.getData()[col]);
    return result;
  }

  public Matrix getGreatestRowNumber() {
    Matrix result = new Matrix(1, cols);
    double[] greatest = new double[cols];
    for (int i = 0; i < cols; i++) {
      greatest[i] = Double.MIN_VALUE;
    }

    forEach(
        (row, col, value) -> {
          if (value > greatest[col]) {
            greatest[col] = value;
            result.data[col] = row;
          }
        });
    return result;
  }

  public int getRows() {
    return rows;
  }

  public int getCols() {
    return cols;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Matrix matrix = (Matrix) o;

    for (int i = 0; i < data.length; i++) {
      if (Math.abs(data[i] - matrix.data[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(rows, cols);
    result = 31 * result + Arrays.hashCode(data);
    return result;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();

    // Berechnen der maximalen Breite jeder Spalte
    int[] maxWidth = new int[cols];
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        int length = String.format(NUMBER_FORMAT, data[row * cols + col]).length();
        if (length > maxWidth[col]) {
          maxWidth[col] = length;
        }
      }
    }

    // Hinzufügen der Rahmenlinien und der Daten
    String rowSeparator =
        "+"
            + Arrays.stream(maxWidth)
                .mapToObj(width -> "-".repeat(width + 2))
                .collect(Collectors.joining("+"))
            + "+\n";

    for (int row = 0; row < rows; row++) {
      sb.append(rowSeparator);
      sb.append("|");
      for (int col = 0; col < cols; col++) {
        String formattedNumber =
            String.format(
                "%" + maxWidth[col] + "s", String.format(NUMBER_FORMAT, data[row * cols + col]));
        sb.append(" ").append(formattedNumber).append(" |");
      }
      sb.append("\n");
    }
    sb.append(rowSeparator);

    return sb.toString();
  }

  public String toString(boolean showValues) {
    if (showValues) {
      return toString();
    } else {
      return "{" + "rows=" + rows + ", cols=" + cols + '}';
    }
  }

  public interface RowColumnProducer {
    double produce(int row, int col, double value);
  }

  public interface Producer {
    double produce(int index);
  }

  public interface IndexValueProducer {
    double produce(int index, double value);
  }

  public interface IndexValueConsumer {
    void consume(int index, double value);
  }

  public interface RowColValueConsumer {
    void consume(int row, int col, double value);
  }

  public interface RowColIndexValueConsumer {
    void consume(int row, int col, int index, double value);
  }
}
