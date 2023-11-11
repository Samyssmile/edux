package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public class MaxPoolingLayer implements Layer {
  private int filterSize;
  private int stride;
  private Matrix lastInput;
  private int[][] maxIndices;

  public MaxPoolingLayer(int filterSize, int stride) {
    this.filterSize = filterSize;
    this.stride = stride;
  }

  @Override
  public Matrix forward(Matrix input) {
    this.lastInput = input;
    int outputHeight = (input.getRows() - filterSize) / stride + 1;
    int outputWidth = (input.getCols() - filterSize) / stride + 1;
    double[][] outputData = new double[outputHeight][outputWidth];
    this.maxIndices = new int[outputHeight * outputWidth][2];

    for (int i = 0; i < outputHeight; i++) {
      for (int j = 0; j < outputWidth; j++) {
        double maxVal = Double.NEGATIVE_INFINITY;
        int maxRow = -1;
        int maxCol = -1;
        for (int fi = 0; fi < filterSize; fi++) {
          for (int fj = 0; fj < filterSize; fj++) {
            double val = input.getData()[i * stride + fi][j * stride + fj];
            if (val > maxVal) {
              maxVal = val;
              maxRow = i * stride + fi;
              maxCol = j * stride + fj;
            }
          }
        }
        outputData[i][j] = maxVal;
        maxIndices[i * outputWidth + j] = new int[] {maxRow, maxCol};
      }
    }

    return new Matrix(outputData, outputHeight, outputWidth);
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    double[][] inputGradientData = new double[lastInput.getRows()][lastInput.getCols()];
    for (int i = 0; i < outputGradient.getRows(); i++) {
      for (int j = 0; j < outputGradient.getCols(); j++) {
        int[] maxIndex = maxIndices[i * outputGradient.getCols() + j];
        inputGradientData[maxIndex[0]][maxIndex[1]] = outputGradient.getData()[i][j];
      }
    }
    return new Matrix(inputGradientData, lastInput.getRows(), lastInput.getCols());
  }
}
