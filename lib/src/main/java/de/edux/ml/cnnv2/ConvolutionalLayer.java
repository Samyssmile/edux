package de.edux.ml.cnnv2;

import de.edux.ml.cnn.layers.Layer;
import de.edux.ml.cnn.math.Matrix3D;

import java.util.Random;

public class ConvolutionalLayer implements Layer {
  private int numFilters;
  private int filterSize;
  private int stride;
  private int padding;
  private int inputDepth;
  private Matrix3D[] filters;
  private Matrix3D lastInput;

  public ConvolutionalLayer(
      int numFilters, int filterSize, int stride, int padding, int inputDepth) {
    this.numFilters = numFilters;
    this.filterSize = filterSize;
    this.stride = stride;
    this.padding = padding;
    this.inputDepth = inputDepth;
    this.filters = new Matrix3D[numFilters];
    initializeFilters();
  }

  private void initializeFilters() {
    Random random = new Random();

    for (int i = 0; i < numFilters; i++) {
      filters[i] = new Matrix3D(inputDepth, filterSize, filterSize);

      for (int depth = 0; depth < inputDepth; depth++) {
        for (int row = 0; row < filterSize; row++) {
          for (int col = 0; col < filterSize; col++) {
            double randomValue = 0.1 * random.nextGaussian();
            filters[i].set(depth, row, col, randomValue);
          }
        }
      }
    }
  }


  @Override
  public Matrix3D forward(Matrix3D input) {
    this.lastInput = input;

    int outputHeight = calculateOutputSize(input.getRows(), filterSize, padding, stride);
    int outputWidth = calculateOutputSize(input.getCols(), filterSize, padding, stride);
    Matrix3D output = new Matrix3D(numFilters, outputHeight, outputWidth);

    for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
      Matrix3D filter = filters[filterIndex];

      for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
          double value = applyFilter(input, filter, y, x);
          output.set(filterIndex, y, x, value);
        }
      }
    }

    return output;
  }

  private int calculateOutputSize(int inputSize, int filterSize, int padding, int stride) {
    return (inputSize - filterSize + 2 * padding) / stride + 1;
  }

  private double applyFilter(Matrix3D input, Matrix3D filter, int startY, int startX) {
    double sum = 0;
    for (int depth = 0; depth < input.getDepth(); depth++) {
      for (int filterY = 0; filterY < filterSize; filterY++) {
        for (int filterX = 0; filterX < filterSize; filterX++) {
          int inputY = startY * stride + filterY - padding;
          int inputX = startX * stride + filterX - padding;

          if (inputY >= 0 && inputY < input.getRows() && inputX >= 0 && inputX < input.getCols()) {
            sum += input.get(depth, inputY, inputX) * filter.get(depth, filterY, filterX);
          }
        }
      }
    }
    return sum;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    Matrix3D inputGradient = new Matrix3D(lastInput.getDepth(), lastInput.getRows(), lastInput.getCols());

    for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
      Matrix3D filter = filters[filterIndex];

      for (int y = 0; y < lastInput.getRows(); y++) {
        for (int x = 0; x < lastInput.getCols(); x++) {
          double gradientSum = 0;
          for (int outY = 0; outY < outputGradient.getRows(); outY++) {
            for (int outX = 0; outX < outputGradient.getCols(); outX++) {
              int filterY = y - outY * stride;
              int filterX = x - outX * stride;

              if (filterY >= 0 && filterY < filterSize && filterX >= 0 && filterX < filterSize) {
                for (int depth = 0; depth < inputGradient.getDepth(); depth++) {
                  gradientSum += filter.get(depth, filterY, filterX) * outputGradient.get(filterIndex, outY, outX);
                  double delta = lastInput.get(depth, y, x) * outputGradient.get(filterIndex, outY, outX);
                  filter.addToValue(depth, filterY, filterX, -learningRate * delta);
                }
              }
            }
          }
          for (int depth = 0; depth < inputGradient.getDepth(); depth++) {
            inputGradient.addToValue(depth, y, x, gradientSum);
          }
        }
      }
    }

    return inputGradient;
  }


}
