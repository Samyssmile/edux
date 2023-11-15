package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;
import java.util.Random;

public class ConvolutionalLayer implements Layer {

  private int numberOfFilters;
  private int filterSize;
  private int stride;
  private int padding;
  private int depth;
  private int inputDepth;
  private int inputRows;
  private int inputCols;

  private double[][][][] filters; // Filter-Array

  // Konstruktor
  public ConvolutionalLayer(
      int numberOfFilters, int filterSize, int stride, int padding, int depth) {
    this.numberOfFilters = numberOfFilters;
    this.filterSize = filterSize;
    this.stride = stride;
    this.padding = padding;
    this.depth = depth;

    // Initialisierung der Filter
    filters = new double[numberOfFilters][depth][filterSize][filterSize];
    Random rand = new Random();
    for (int i = 0; i < numberOfFilters; i++) {
      for (int j = 0; j < depth; j++) {
        for (int k = 0; k < filterSize; k++) {
          for (int l = 0; l < filterSize; l++) {
            filters[i][j][k][l] = rand.nextGaussian();
          }
        }
      }
    }
  }

  @Override
  public Matrix3D forward(Matrix3D input) {
    if (this.inputDepth == 0) {
      this.inputDepth = input.getDepth();
      this.inputRows = input.getRows();
      this.inputCols = input.getCols();
    }
    int outputDepth = numberOfFilters;
    int outputRows = (input.getRows() - filterSize + 2 * padding) / stride + 1;
    int outputCols = (input.getCols() - filterSize + 2 * padding) / stride + 1;

    Matrix3D output = new Matrix3D(outputDepth, outputRows, outputCols);

    for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
      for (int row = 0; row < outputRows; row++) {
        for (int col = 0; col < outputCols; col++) {
          double sum = 0.0;
          for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
              for (int k = 0; k < depth; k++) {
                int inputRow = row * stride + i - padding;
                int inputCol = col * stride + j - padding;
                double inputValue =
                    (inputRow >= 0
                            && inputRow < input.getRows()
                            && inputCol >= 0
                            && inputCol < input.getCols())
                        ? input.get(k, inputRow, inputCol)
                        : 0.0;
                sum += inputValue * filters[filterIndex][k][i][j];
              }
            }
          }
          output.set(filterIndex, row, col, sum);
        }
      }
    }

    return output;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    Matrix3D inputGradient = new Matrix3D(inputDepth, inputRows, inputCols);

    // Gradientenberechnung für jeden Filter
    for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
      for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
          for (int k = 0; k < depth; k++) {
            double filterGradient = 0.0;
            for (int row = 0; row < outputGradient.getRows(); row++) {
              for (int col = 0; col < outputGradient.getCols(); col++) {
                int inputRow = row * stride + i - padding;
                int inputCol = col * stride + j - padding;
                if (inputRow >= 0
                    && inputRow < inputGradient.getRows()
                    && inputCol >= 0
                    && inputCol < inputGradient.getCols()) {
                  filterGradient +=
                      outputGradient.get(filterIndex, row, col)
                          * inputGradient.get(k, inputRow, inputCol);
                }
              }
            }
            // Aktualisierung des Filters
            filters[filterIndex][k][i][j] -= learningRate * filterGradient;
          }
        }
      }
    }

    for (int k = 0; k < depth; k++) {
      for (int row = 0; row < inputGradient.getRows(); row++) {
        for (int col = 0; col < inputGradient.getCols(); col++) {
          double sum = 0.0;
          for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
            for (int i = 0; i < filterSize; i++) {
              for (int j = 0; j < filterSize; j++) {
                int outRow = (row + padding - i) / stride;
                int outCol = (col + padding - j) / stride;

                if (outRow >= 0
                    && outRow < outputGradient.getRows()
                    && outCol >= 0
                    && outCol < outputGradient.getCols()) {
                  sum +=
                      outputGradient.get(filterIndex, outRow, outCol)
                          * filters[filterIndex][k][i][j];
                }
              }
            }
          }
          inputGradient.set(k, row, col, sum);
        }
      }
    }

    return inputGradient;
  }

  public void setFilters(double[][][][] doubles) {
    this.filters = doubles;
  }

  // Weitere Methoden für die interne Funktionalität des ConvolutionalLayer
}
