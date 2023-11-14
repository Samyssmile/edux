package de.edux.ml.cnnv2;

import de.edux.ml.cnn.layers.Layer;
import de.edux.ml.cnn.math.Matrix3D;

public class DenseLayer implements Layer {
  private Matrix3D weights;
  private Matrix3D biases;
  private Matrix3D input;
  private Matrix3D outputGradient;

  public DenseLayer(int inputSize, int outputSize, int depth) {
    // Initialisiere die Gewichtsmatrix und die Bias-Matrix
    this.weights = Matrix3D.random(depth, outputSize, inputSize); // depth, rows, cols
    this.biases = new Matrix3D(depth, outputSize, 1); // depth, rows, cols
  }

  @Override
  public Matrix3D forward(Matrix3D input) {
    this.input = input;
    // Implementiere die Forward-Pass Operation: output = (input * weights) + biases
    Matrix3D output =
        input.dot(weights.transpose()); // Achte darauf, die Matrizen richtig zu transponieren
    output = output.add(biases.reshape(1, output.getRows(), output.getCols()));

    return output;
  }

  @Override
  public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
    // Berechnung des Gradienten in Bezug auf die Eingabe
    Matrix3D inputGradient = outputGradient.dot(weights);

    // Berechnung des Gradienten in Bezug auf die Gewichte
    Matrix3D weightGradient = outputGradient.transpose().dot(input);

    // Anpassung des outputGradient für die Biases
    Matrix3D biasGradient =
        outputGradient.sum(1); // Summieren über die Spalten, um die korrekte Form zu erhalten

    // Update weights and biases
    weights = weights.subtract(weightGradient.multiply(learningRate));
    biases = biases.subtract(biasGradient.multiply(learningRate));

    return inputGradient;
  }
}
