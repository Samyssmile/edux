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
    // Initialisieren des Eingangsgradienten
    Matrix3D inputGradient =
        new Matrix3D(lastInput.getDepth(), lastInput.getRows(), lastInput.getCols());

    // Rückwärtspropagation des Max Pooling Layers
    for (int d = 0; d < lastInput.getDepth(); d++) {
      for (int row = 0; row < lastInput.getRows(); row += stride) {
        for (int col = 0; col < lastInput.getCols(); col += stride) {
          // Finden des Maximums im Pooling-Fenster
          double maxVal = Double.NEGATIVE_INFINITY;
          int maxRow = -1, maxCol = -1;
          for (int pRow = 0; pRow < poolSize; pRow++) {
            for (int pCol = 0; pCol < poolSize; pCol++) {
              int curRow = row + pRow;
              int curCol = col + pCol;
              if (curRow < lastInput.getRows() && curCol < lastInput.getCols()) {
                double val = lastInput.get(d, curRow, curCol);
                if (val > maxVal) {
                  maxVal = val;
                  maxRow = curRow;
                  maxCol = curCol;
                }
              }
            }
          }
          // Zuweisen des Gradienten nur zum Max-Element
          if (maxRow != -1 && maxCol != -1) {
            int outRow = row / stride;
            int outCol = col / stride;
            inputGradient.set(d, maxRow, maxCol, outputGradient.get(d, outRow, outCol));
          }
        }
      }
    }
    return inputGradient;
  }
}
