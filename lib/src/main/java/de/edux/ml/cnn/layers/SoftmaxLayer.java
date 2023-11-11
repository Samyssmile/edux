package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public class SoftmaxLayer implements Layer {
  private Matrix output;

  @Override
  public Matrix forward(Matrix input) {
    output = applySoftmax(input);
    return output;
  }

  private Matrix applySoftmax(Matrix input) {
    double max = max(input);
    double sum = 0.0;
    double[][] outputData = new double[input.getRows()][input.getCols()];

    for (int i = 0; i < input.getRows(); i++) {
      for (int j = 0; j < input.getCols(); j++) {
        outputData[i][j] = Math.exp(input.getData()[i][j] - max);
        sum += outputData[i][j];
      }
    }

    for (int i = 0; i < input.getRows(); i++) {
      for (int j = 0; j < input.getCols(); j++) {
        outputData[i][j] /= sum;
      }
    }

    return new Matrix(outputData, input.getRows(), input.getCols());
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    return outputGradient;
  }

  // Helper method to find the maximum value in the input matrix
  private double max(Matrix matrix) {
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < matrix.getRows(); i++) {
      for (int j = 0; j < matrix.getCols(); j++) {
        if (matrix.getData()[i][j] > max) {
          max = matrix.getData()[i][j];
        }
      }
    }
    return max;
  }
}
