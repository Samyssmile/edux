package de.edux.core.math;

/** The IMatrixProduct interface defines a method for multiplying two matrices. */
public interface IMatrixProduct {

  /**
   * Multiplies two matrices together and returns the result as a new matrix.
   *
   * @param matrixA The first matrix to be multiplied, represented as a 2D array of doubles.
   * @param matrixB The second matrix to be multiplied, represented as a 2D array of doubles.
   * @return The product of matrixA and matrixB, represented as a 2D array of doubles.
   * @throws IllegalArgumentException If the matrices cannot be multiplied due to incompatible
   *     dimensions.
   */
  double[][] multiply(double[][] matrixA, double[][] matrixB);
}
