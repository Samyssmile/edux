package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public class ReLuLayer implements Layer {
  private Matrix3D lastInput;

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.lastInput = input;
    return input.applyReLU();
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    return lastInput.applyReLUBackward(outputGradient);
  }
}
