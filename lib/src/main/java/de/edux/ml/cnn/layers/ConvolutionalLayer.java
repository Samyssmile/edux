package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

/*TODO padding einbauen.
 *  Die forward-Methode sieht angemessen aus, sie wendet die Filter auf das Eingabebild an. Etwas zu beachten ist,
 * dass diese Implementierung keine Padding berücksichtigt. Dies bedeutet, dass die Ausgabegröße der Schicht kleiner sein wird als die Eingabegröße.*/
public class ConvolutionalLayer implements Layer {
  private Matrix[] filters;
  private int numFilters;
  private int filterSize;
  private Matrix input; // Store the input to use in the backward pass

  public ConvolutionalLayer(int numFilters, int filterSize) {
    this.numFilters = numFilters;
    this.filterSize = filterSize;
    this.filters = new Matrix[numFilters];
    for (int i = 0; i < numFilters; i++) {
      // Initialize each filter with random weights
      this.filters[i] = Matrix.random(filterSize, filterSize);
    }
  }

  @Override
  public Matrix forward(Matrix input) {
    this.input = input; // Store the input
    // Assuming a single channel input. For multi-channel, you'll need to adjust this.
    int outputHeight = input.getRows() - filterSize + 1;
    int outputWidth = input.getCols() - filterSize + 1;
    double[][] outputData = new double[outputHeight][outputWidth];

    for (int f = 0; f < numFilters; f++) {
      for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
          outputData[i][j] = applyFilter(input, filters[f], i, j);
        }
      }
      // Assuming outputs are stored or combined here
    }
    // Assuming outputs are combined into a single Matrix
    return new Matrix(outputData, outputHeight, outputWidth);
  }

  private double applyFilter(Matrix input, Matrix filter, int row, int col) {
    double sum = 0;
    for (int i = 0; i < filterSize; i++) {
      for (int j = 0; j < filterSize; j++) {
        sum += input.getData()[row + i][col + j] * filter.getData()[i][j];
      }
    }
    return sum;
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    double[][] inputGradientData = new double[input.getRows()][input.getCols()];
    Matrix inputGradient = new Matrix(inputGradientData, input.getRows(), input.getCols());

    for (int f = 0; f < numFilters; f++) {
      double[][] filterGradientData = new double[filterSize][filterSize];
      Matrix filterGradient = new Matrix(filterGradientData, filterSize, filterSize);

      for (int i = 0; i < outputGradient.getRows(); i++) {
        for (int j = 0; j < outputGradient.getCols(); j++) {
          // Calculate the gradient for the filter
          filterGradient = updateFilterGradient(filterGradient, outputGradient, i, j, f);
          // Update the input gradient
          inputGradient = updateInputGradient(inputGradient, outputGradient, filters[f], i, j);
        }
      }
      // Update filter weights
      filters[f] = filters[f].subtract(filterGradient.multiply(learningRate));
    }
    return inputGradient;
  }

  private Matrix updateFilterGradient(
      Matrix filterGradient, Matrix outputGradient, int row, int col, int filterIndex) {
    double[][] gradientData = filterGradient.getData();
    double outputGrad = outputGradient.getData()[row][col];

    for (int i = 0; i < filterSize; i++) {
      for (int j = 0; j < filterSize; j++) {
        gradientData[i][j] += outputGrad * input.getData()[row + i][col + j];
      }
    }
    return new Matrix(gradientData, filterSize, filterSize);
  }

  private Matrix updateInputGradient(
      Matrix inputGradient, Matrix outputGradient, Matrix filter, int row, int col) {
    double[][] inputGradientData = inputGradient.getData();
    double outputGrad = outputGradient.getData()[row][col];

    for (int i = 0; i < filterSize; i++) {
      for (int j = 0; j < filterSize; j++) {
        inputGradientData[row + i][col + j] += outputGrad * filter.getData()[i][j];
      }
    }
    return new Matrix(inputGradientData, inputGradient.getRows(), inputGradient.getCols());
  }
}
