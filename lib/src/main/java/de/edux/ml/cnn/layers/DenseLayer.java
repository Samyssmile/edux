package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public class DenseLayer implements Layer {
  private Matrix3D weights;
  private Matrix3D biases;
  private Matrix3D input;
  private Matrix3D outputGradient;

  public DenseLayer(int inputSize, int outputSize) {
    // Gewichte mit Dimensionen 1, outputSize, inputSize initialisieren
    this.weights = Matrix3D.random(1, outputSize, inputSize); // Gewichte
    this.biases = new Matrix3D(1, 1, outputSize); // Biases
  }

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.input = input;
    Matrix3D output = input.dot(this.weights.transpose()).add(this.biases);
    return output;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    this.outputGradient = outputGradient;

    Matrix3D weightGradient = outputGradient.dot(this.input);
    // Gradient bezüglich der Eingabe
    Matrix3D inputGradient = this.weights.dot(outputGradient);

    // Aktualisieren der Gewichte und Biases
    this.weights = this.weights.subtract(weightGradient.multiply(learningRate));
    this.biases = this.biases.subtract(outputGradient.sumColumns().multiply(learningRate));

    return inputGradient;
  }
}
