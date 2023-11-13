package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public class ConvolutionalLayer implements Layer {
  private Matrix3D[] filters;
  private int numFilters;
  private int filterSize;
  private int stride;
  private int padding;
  private Matrix3D input;

  public ConvolutionalLayer(
      int numFilters, int filterSize, int stride, int padding, int inputDepth) {
    this.numFilters = numFilters;
    this.filterSize = filterSize;
    this.stride = stride;
    this.padding = padding;
    this.filters = new Matrix3D[numFilters];

    for (int i = 0; i < numFilters; i++) {
      this.filters[i] = Matrix3D.random(inputDepth, filterSize, filterSize);
    }
  }

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.input = input;
    // Padding
    int outputHeight = (input.getRows() - filterSize + 2 * padding) / stride + 1;
    int outputWidth = (input.getCols() - filterSize + 2 * padding) / stride + 1;
    Matrix3D output = new Matrix3D(numFilters, outputHeight, outputWidth);

    // Führen Sie die Faltungsoperation für jeden Filter durch
    for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
      for (int row = 0; row < outputHeight; row++) {
        for (int col = 0; col < outputWidth; col++) {
          double sum = 0.0;

          for (int filterRow = 0; filterRow < filterSize; filterRow++) {
            for (int filterCol = 0; filterCol < filterSize; filterCol++) {
              int inputRow = row * stride + filterRow - padding;
              int inputCol = col * stride + filterCol - padding;

              if (inputRow >= 0
                  && inputRow < input.getRows()
                  && inputCol >= 0
                  && inputCol < input.getCols()) {
                // Führen Sie die elementweise Multiplikation durch und summieren Sie die Ergebnisse
                sum +=
                    input.get(0, inputRow, inputCol)
                        * filters[filterIndex].get(0, filterRow, filterCol);
              }
            }
          }

          // Zuweisen des Summenwerts zur entsprechenden Position in der Ausgabematrix
          output.set(filterIndex, row, col, sum);
        }
      }
    }

    return output;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    Matrix3D inputGradient = new Matrix3D(input.getDepth(), input.getRows(), input.getCols());

    // Aktualisieren der Filter und Berechnen des Eingangsgradienten
    for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
      Matrix3D filterGradient = computeGradientForFilter(outputGradient, filterIndex);

      Matrix3D filterGradientMulLearingrate = filterGradient.multiply(learningRate);
      // Aktualisieren der Filtergewichte
      filters[filterIndex] = filters[filterIndex].subtract(filterGradientMulLearingrate);

      // Berechnen des Eingangsgradienten
      for (int row = 0; row < input.getRows(); row++) {
        for (int col = 0; col < input.getCols(); col++) {
          for (int filterRow = 0; filterRow < filterSize; filterRow++) {
            for (int filterCol = 0; filterCol < filterSize; filterCol++) {
              int outRow = (row + padding - filterRow) / stride;
              int outCol = (col + padding - filterCol) / stride;

              if (outRow >= 0
                  && outRow < outputGradient.getRows()
                  && outCol >= 0
                  && outCol < outputGradient.getCols()) {
                double value =
                    inputGradient.get(0, row, col)
                        + filters[filterIndex].get(0, filterRow, filterCol)
                            * outputGradient.get(filterIndex, outRow, outCol);
                inputGradient.set(0, row, col, value);
              }
            }
          }
        }
      }
    }

    return inputGradient;
  }

  private Matrix3D computeGradientForFilter(Matrix3D outputGradient, int filterIndex) {
    int filterDepth = this.input.getDepth();
    Matrix3D filterGradient = new Matrix3D(filterDepth, filterSize, filterSize);

    for (int d = 0; d < filterDepth; d++) {
      for (int row = 0; row < filterSize; row++) {
        for (int col = 0; col < filterSize; col++) {
          double sum = 0.0;

          for (int outRow = 0; outRow < outputGradient.getRows(); outRow++) {
            for (int outCol = 0; outCol < outputGradient.getCols(); outCol++) {
              int inputRow = outRow * this.stride + row - this.padding;
              int inputCol = outCol * this.stride + col - this.padding;

              if (inputRow >= 0
                  && inputRow < this.input.getRows()
                  && inputCol >= 0
                  && inputCol < this.input.getCols()) {
                sum +=
                    this.input.get(d, inputRow, inputCol)
                        * outputGradient.get(filterIndex, outRow, outCol);
              }
            }
          }

          filterGradient.set(d, row, col, sum);
        }
      }
    }

    return filterGradient;
  }
}
