package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public class MaxPoolingLayer implements Layer {
  private int poolSize;
  private int stride;
  private int[][] maxIndices; // Zum Speichern der Indizes der Max-Werte für den Backward-Pass
  private Matrix3D lastInput;

  public MaxPoolingLayer(int poolSize, int stride) {
    this.poolSize = poolSize;
    this.stride = stride;
  }

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.lastInput = input;
    int outputDepth = input.getDepth();
    int outputRows = (input.getRows() - poolSize) / stride + 1;
    int outputCols = (input.getCols() - poolSize) / stride + 1;
    Matrix3D output = new Matrix3D(outputDepth, outputRows, outputCols);

    maxIndices = new int[outputDepth][outputRows * outputCols];

    for (int d = 0; d < outputDepth; d++) {
      for (int row = 0; row < outputRows; row++) {
        for (int col = 0; col < outputCols; col++) {
          double max = -Double.MAX_VALUE;
          int maxIndex = -1;
          for (int i = 0; i < poolSize; i++) {
            for (int j = 0; j < poolSize; j++) {
              int rIndex = row * stride + i;
              int cIndex = col * stride + j;
              double value = input.get(d, rIndex, cIndex);
              if (value > max) {
                max = value;
                maxIndex = rIndex * input.getCols() + cIndex;
              }
            }
          }
          output.set(d, row, col, max);
          maxIndices[d][row * outputCols + col] = maxIndex;
        }
      }
    }

    return output;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    Matrix3D inputGradient =
        new Matrix3D(this.lastInput.getDepth(), this.lastInput.getRows(), this.lastInput.getCols());

    for (int d = 0; d < inputGradient.getDepth(); d++) {
      for (int i = 0; i < maxIndices[d].length; i++) {
        int index = maxIndices[d][i];
        int row = index / this.lastInput.getCols();
        int col = index % this.lastInput.getCols();
        inputGradient.set(
            d,
            row,
            col,
            outputGradient.get(d, i / outputGradient.getCols(), i % outputGradient.getCols()));
      }
    }

    return inputGradient;
  }
}
