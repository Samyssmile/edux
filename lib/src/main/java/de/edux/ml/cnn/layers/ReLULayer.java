package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public class ReLULayer implements Layer {
  private Matrix input;

  @Override
  public Matrix forward(Matrix input) {
    this.input = input;
    double[][] outputData = new double[input.getRows()][input.getCols()];

    for (int i = 0; i < input.getRows(); i++) {
      for (int j = 0; j < input.getCols(); j++) {
        outputData[i][j] = Math.max(0, input.getData()[i][j]);
      }
    }

    return new Matrix(outputData, input.getRows(), input.getCols());
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    double[][] inputGradientData = new double[input.getRows()][input.getCols()];

    for (int i = 0; i < input.getRows(); i++) {
      for (int j = 0; j < input.getCols(); j++) {
        inputGradientData[i][j] = input.getData()[i][j] > 0 ? outputGradient.getData()[i][j] : 0;
      }
    }

    return new Matrix(inputGradientData, input.getRows(), input.getCols());
  }
}
