package de.edux.util.math;

public interface ConcurrentMatrixMultiplication {

  /**
   * Multiplies two matrices and returns the resulting matrix.
   *
   * @param a The first matrix.
   * @param b The second matrix.
   * @return The product of the two matrices.
   * @throws IllegalArgumentException If the matrices cannot be multiplied due to incompatible
   *     dimensions.
   */
  double[][] multiplyMatrices(double[][] a, double[][] b)
      throws IllegalArgumentException, IncompatibleDimensionsException;
}
