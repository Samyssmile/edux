package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public class SoftmaxLayer implements Layer {

  private Matrix3D output;
  private Matrix3D inputGradient;

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.output = applySoftmax(input);
    return this.output;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    this.inputGradient =
        this.output.multiplyElementWise(outputGradient.subtract(this.output.sumColumns()));
    return this.inputGradient;
  }

  private Matrix3D applySoftmax(Matrix3D input) {
    Matrix3D softmaxOutput = new Matrix3D(input.getDepth(), input.getRows(), input.getCols());

    for (int d = 0; d < input.getDepth(); d++) {
      double max = findMaxInColumn(input, d);
      double sum = 0.0;

      // Berechnung des Nenners der Softmax-Funktion
      for (int c = 0; c < input.getCols(); c++) {
        sum += Math.exp(input.get(d, 0, c) - max);
      }

      // Berechnung der Softmax-Werte
      for (int c = 0; c < input.getCols(); c++) {
        softmaxOutput.set(d, 0, c, Math.exp(input.get(d, 0, c) - max) / sum);
      }
    }

    // sum all values of all columns
    for (int d = 0; d < input.getDepth(); d++) {
      double sum = 0.0;
      for (int c = 0; c < input.getCols(); c++) {
        sum += softmaxOutput.get(d, 0, c);
      }
      System.out.println("sum of all values of all columns: " + sum);
    }
    return softmaxOutput;
  }

  private double findMaxInColumn(Matrix3D matrix, int depth) {
    double max = Double.NEGATIVE_INFINITY;
    for (int c = 0; c < matrix.getCols(); c++) {
      if (matrix.get(depth, 0, c) > max) {
        max = matrix.get(depth, 0, c);
      }
    }
    return max;
  }
}
