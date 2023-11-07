package de.edux.core.math;

/**
 * The IMatrixVectorProduct interface specifies the operation for multiplying a matrix by a vector.
 */
public interface IMatrixVectorProduct {

  /**
   * Multiplies a matrix by a vector and returns the result as a new vector.
   *
   * <p>This method takes a two-dimensional array representing a matrix and a one-dimensional array
   * representing a vector, performs the matrix-vector multiplication, and returns the resulting
   * vector as a one-dimensional array.
   *
   * @param matrix The matrix to be multiplied, represented as a 2D array of doubles. The number of
   *     columns in the matrix must match the size of the vector.
   * @param vector The vector to be multiplied, represented as a 1D array of doubles.
   * @return The product of the matrix and vector multiplication, represented as a 1D array of
   *     doubles.
   * @throws IllegalArgumentException If the number of columns in the matrix does not match the size
   *     of the vector, indicating that the matrix and vector cannot be multiplied due to
   *     incompatible dimensions.
   */
  double[] multiply(double[][] matrix, double[] vector);
}
