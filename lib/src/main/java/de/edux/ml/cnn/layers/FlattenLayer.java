package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public class FlattenLayer implements Layer {

  private int lastInputDepth;
  private int lastInputRows;
  private int lastInputCols;

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.lastInputDepth = input.getDepth();
    this.lastInputRows = input.getRows();
    this.lastInputCols = input.getCols();

    return input.flatten();
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    return outputGradient.reshapeBack(lastInputDepth, lastInputRows, lastInputCols);
  }
}
