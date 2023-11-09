package de.edux.core.math.matrix.parallel;

import de.edux.core.math.IMatrixArithmetic;
import de.edux.core.math.IMatrixProduct;
import de.edux.core.math.IMatrixVectorProduct;
import de.edux.core.math.matrix.parallel.operations.MatrixProduct;
import de.edux.core.math.matrix.parallel.operations.MatrixVectorProduct;

public class MatrixArithmetic implements IMatrixArithmetic {

  private final IMatrixProduct matrixProduct;
  private final IMatrixVectorProduct matrixVectorProduct;

  public MatrixArithmetic() {
    this.matrixProduct = new MatrixProduct();
    this.matrixVectorProduct = new MatrixVectorProduct();
  }

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {

    if (matrixA == null || matrixB == null) {
      throw new IllegalArgumentException("Matrices must not be null.");
    }
    if (matrixA.length == 0 || matrixB.length == 0) {
      throw new IllegalArgumentException("Matrices must not be empty.");
    }
    if (matrixA[0].length != matrixB.length) {
      throw new IllegalArgumentException("Matrix A columns must match Matrix B rows.");
    }

    return matrixProduct.multiply(matrixA, matrixB);
  }

  @Override
  public double[] multiply(double[][] matrix, double[] vector) {

    if (matrix.length == 0 || matrix[0].length == 0 || vector.length == 0) {
      throw new IllegalArgumentException("Matrix and vector must not be empty.");
    }
    if (matrix[0].length != vector.length) {
      throw new IllegalArgumentException("Matrix columns and vector size do not match.");
    }

    return matrixVectorProduct.multiply(matrix, vector);
  }
}
