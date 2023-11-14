package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public class DenseLayer implements Layer {
  private Matrix3D weights;
  private Matrix3D biases;
  private Matrix3D input;

  public DenseLayer(int inputSize, int outputSize, int depth) {
    this.weights = Matrix3D.randomHe(depth, outputSize, inputSize); // Gewichte
    this.biases = new Matrix3D(depth, 1, outputSize); // Biases
  }

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.input = input;
    Matrix3D output = input.dot(this.weights.transpose()).add(this.biases);
    return output;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    // Berechnung des Gradienten für die Gewichte
    Matrix3D weightGradient = this.input.transpose().dot(outputGradient);

    // Transponieren des Gewichtsgradienten, um die Dimensionen anzupassen
    Matrix3D transposedWeightGradient = weightGradient.transpose();

    // Berechnung des Gradienten für die Biases
    Matrix3D biasGradient = outputGradient.sum(0);

    // Aktualisierung der Gewichte und Biases
    this.weights = this.weights.subtract(transposedWeightGradient.multiply(learningRate));
    this.biases = this.biases.subtract(biasGradient.multiply(learningRate));

    // Berechnung des Eingangsgradienten für die vorherige Schicht
    Matrix3D inputGradient = outputGradient.dot(this.weights);

    return inputGradient;
  }
}
