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
    return matrixProduct.multiply(matrixA, matrixB);
  }

  @Override
  public double[] multiply(double[][] matrix, double[] vector) {
    return matrixVectorProduct.multiply(matrix, vector);
  }
}
