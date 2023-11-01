package de.edux.util.math;

public interface MatrixOperations {
  /**
   * Adds two matrices and returns the resulting matrix.
   *
   * @param a The first matrix.
   * @param b The second matrix.
   * @return The sum of the two matrices.
   * @throws IllegalArgumentException If the matrices are not of the same dimension.
   */
  double[][] addMatrices(double[][] a, double[][] b) throws IllegalArgumentException;

  /**
   * Subtracts matrix b from matrix a and returns the resulting matrix.
   *
   * @param a The first matrix.
   * @param b The second matrix.
   * @return The result of a - b.
   * @throws IllegalArgumentException If the matrices are not of the same dimension.
   */
  double[][] subtractMatrices(double[][] a, double[][] b) throws IllegalArgumentException;

  /**
   * Transposes the given matrix and returns the resulting matrix.
   *
   * @param a The matrix to transpose.
   * @return The transposed matrix.
   */
  double[][] transposeMatrix(double[][] a);

  /**
   * Inverts the given matrix and returns the resulting matrix.
   *
   * @param a The matrix to invert.
   * @return The inverted matrix.
   * @throws IllegalArgumentException If the matrix is not invertible.
   */
  double[][] invertMatrix(double[][] a) throws IllegalArgumentException;

  /**
   * Calculates and returns the determinant of the given matrix.
   *
   * @param a The matrix.
   * @return The determinant of the matrix.
   * @throws IllegalArgumentException If the matrix is not square.
   */
  double determinant(double[][] a) throws IllegalArgumentException;
}
