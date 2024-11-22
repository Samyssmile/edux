package de.edux.ml.mlp.core.tensor;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
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
    if (data== null){
        System.out.println("HOLD");
    }
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

    Matrix temp = new Matrix(this.cols, this.rows);
    if (temp.data == null){
        System.out.println("HOLD");
    }
    if (values == null){
        System.out.println("HOLD");
    }
    temp.setData(values);
    Matrix transposed = temp.transpose();
    if (transposed.data == null){
        System.out.println("HOLD");
    }
    data = transposed.data;
  }

  private void setData(double[] values) {
    if (values != null){
      System.arraycopy(values, 0, data, 0, values.length);
    }else{
      throw new IllegalArgumentException("Values Array must not be null");
    }
  }

  public Matrix(double[][] values) {
    this.rows = values.length;
    this.cols = values[0].length;
    if (this.data == null){
      System.out.println("HOLD");
    }
    this.data = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
      if (this.data == null){
        System.out.println("HOLD");
      }
      System.arraycopy(values[i], 0, this.data, i * cols, cols);
    }
  }

  public Matrix multiplyParallel(double rate) {
    Matrix result = new Matrix(this.rows, this.cols);

    IntStream.range(0, this.rows)
        .parallel()
        .forEach(
            i -> {
              if (this.data == null){
                System.out.println("HOLD");
              }
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

    if (this.data == null){
      System.out.println("HOLD");
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

    if (this.data == null){
      System.out.println("HOLD");
    }
    Matrix result = new Matrix(this.rows, this.cols);
    for (int i = 0; i < this.data.length; i++) {

      double termOne = this.data[i];

      double termTwo = matrix.getData()[i];
      result.data[i] = termOne - termTwo;

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
        if (result.data == null){
          System.out.println("HOLD");
        }
        if (data == null){
          System.out.println("HOLD");
        }


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
          if (this.data == null){
            System.out.println("HOLD");
          }
          // Addiere den Bias, wenn die zweite Matrix eine Spaltenmatrix ist
          result.data[row * cols + col] = this.data[row * cols + col] + other.data[row];
        } else {
          if (this.data == null){
            System.out.println("HOLD");
          }
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
    //alternatie parallel soultion
    int index = 0;

    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        consumer.consume(row, col, data[index++]);
      }
    }

/*    IntStream.range(0, rows)
        .parallel()
        .forEach(
            row -> {
              for (int col = 0; col < cols; col++) {
                consumer.consume(row, col, data[row * cols + col]);
              }
            });*/

    

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

  public static Matrix random(int rows, int cols) {
    Matrix matrix = new Matrix(rows, cols);
    Random random = new Random();
    double standardDeviation = Math.sqrt(2.0 / (rows * cols));

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        matrix.set(i, j, random.nextGaussian() * standardDeviation);
      }
    }

    return matrix;
  }

  public static Matrix convolve(double[][][] image, double[][][][] filters, double[] biases,
                                int filterSize, int numFilters, int outputHeight, int outputWidth) {
    int channels = image.length;

    // Vorbereitung der Datenmatrix (im2col)
    int dataMatrixRows = filterSize * filterSize * channels;
    int dataMatrixCols = outputHeight * outputWidth;
    double[][] dataMatrix = new double[dataMatrixRows][dataMatrixCols];

    int columnIndex = 0;
    for (int i = 0; i < outputHeight; i++) {
      for (int j = 0; j < outputWidth; j++) {
        int idx = 0;
        for (int c = 0; c < channels; c++) {
          for (int fi = 0; fi < filterSize; fi++) {
            for (int fj = 0; fj < filterSize; fj++) {
              dataMatrix[idx++][columnIndex] = image[c][i + fi][j + fj];
            }
          }
        }
        columnIndex++;
      }
    }

    // Vorbereitung der Filtermatrix
    double[][] filtersMatrix = new double[numFilters][dataMatrixRows];
    for (int f = 0; f < numFilters; f++) {
      int idx = 0;
      for (int c = 0; c < channels; c++) {
        for (int fi = 0; fi < filterSize; fi++) {
          for (int fj = 0; fj < filterSize; fj++) {
            filtersMatrix[f][idx++] = filters[f][c][fi][fj];
          }
        }
      }
    }

    // Parallele Matrixmultiplikation
    double[][] outputMatrix = new double[numFilters][dataMatrixCols];

    // Parallelisierung über Filter und Ausgabespalten
    IntStream.range(0, numFilters).parallel().forEach(f -> {
      double[] filter = filtersMatrix[f];
      double bias = biases[f];
      IntStream.range(0, dataMatrixCols).parallel().forEach(col -> {
        double sum = 0;
        for (int k = 0; k < dataMatrixRows; k++) {
          sum += filter[k] * dataMatrix[k][col];
        }
        sum += bias;
        outputMatrix[f][col] = sum;
      });
    });

    // Ausgabe vorbereiten
    Matrix output = new Matrix(numFilters * outputHeight * outputWidth, 1);
    int outputIndex = 0;
    for (int f = 0; f < numFilters; f++) {
      for (int col = 0; col < dataMatrixCols; col++) {
        output.set(outputIndex++, 0, outputMatrix[f][col]);
      }
    }

    return output;
  }


  public Matrix softmax() {
    // Berechne das Maximum pro Spalte (für Stabilität)
    Matrix maxColumn = new Matrix(1, cols, col -> Arrays.stream(getColumn(col)).max().orElse(0));

    // Initialisiere das Ergebnis und berechne stabilisierte Exponentialwerte
    Matrix result = new Matrix(rows, cols);
    double[] expSumPerCol = new double[cols]; // Speichert die Spaltensummen

    for (int col = 0; col < cols; col++) {
      double max = maxColumn.get(0, col);
      for (int row = 0; row < rows; row++) {
        double stabilizedExp = Math.exp(get(row, col) - max);
        result.set(row, col, stabilizedExp);
        expSumPerCol[col] += stabilizedExp; // Summiere direkt während der Berechnung
      }
    }

    // Normiere die Werte
    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < rows; row++) {
        double normalized = result.get(row, col) / expSumPerCol[col];
        result.set(row, col, normalized);
      }
    }
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
/*
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
    sb.append(rowSeparator);*/

    return sb.toString();
  }

  public String toString(boolean showValues) {
    if (showValues) {
      return toString();
    } else {
      return "{" + "rows=" + rows + ", cols=" + cols + '}';
    }
  }

  public double[] getColumn(int b) {
    double[] result = new double[rows];
    for (int i = 0; i < rows; i++) {
      result[i] = data[i * cols + b];
    }
    return result;
  }

  public void setColumn(int b, double[] inputGradColumn) {
    for (int i = 0; i < rows; i++) {
      data[i * cols + b] = inputGradColumn[i];
    }
  }

  public boolean hasNaN() {
    for(int i = 0; i < data.length; i++) {
      if (Double.isNaN(data[i])) {
        System.err.println("NaN detected at index " + i);
        return true;
      }
    }
    return false;
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
