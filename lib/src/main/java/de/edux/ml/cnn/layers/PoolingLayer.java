package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public class PoolingLayer implements Layer {
  private int filterSize;
  private int stride;
  private Matrix lastInput;
  private Matrix lastOutput;

  public PoolingLayer(int filterSize, int stride) {
    this.filterSize = filterSize;
    this.stride = stride;
  }

  @Override
  public Matrix forward(Matrix input) {
    this.lastInput = input;
    int outputHeight = (input.getRows() - filterSize) / stride + 1;
    int outputWidth = (input.getCols() - filterSize) / stride + 1;
    double[][] outputData = new double[outputHeight][outputWidth];

    for (int i = 0; i < outputHeight; i++) {
      for (int j = 0; j < outputWidth; j++) {
        outputData[i][j] = applyMaxPooling(input, i, j);
      }
    }

    this.lastOutput = new Matrix(outputData, outputHeight, outputWidth);
    return this.lastOutput;
  }

  private double applyMaxPooling(Matrix input, int row, int col) {
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < filterSize; i++) {
      for (int j = 0; j < filterSize; j++) {
        double val = input.getData()[row * stride + i][col * stride + j];
        if (val > max) {
          max = val;
        }
      }
    }
    return max;
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    double[][] inputData = new double[lastInput.getRows()][lastInput.getCols()];
    Matrix inputGradient = new Matrix(inputData, lastInput.getRows(), lastInput.getCols());

    for (int i = 0; i < lastOutput.getRows(); i++) {
      for (int j = 0; j < lastOutput.getCols(); j++) {
        double maxVal = lastOutput.getData()[i][j];
        for (int di = 0; di < filterSize; di++) {
          for (int dj = 0; dj < filterSize; dj++) {
            if (lastInput.getData()[i * stride + di][j * stride + dj] == maxVal) {
              inputData[i * stride + di][j * stride + dj] = outputGradient.getData()[i][j];
              break;
            }
          }
        }
      }
    }

    return inputGradient;
  }
}
