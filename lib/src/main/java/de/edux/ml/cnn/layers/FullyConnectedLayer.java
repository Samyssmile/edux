package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public class FullyConnectedLayer implements Layer {
  private final int outputSize;
  private Matrix weights;
  private Matrix bias;
  private Matrix input;
  private Matrix output;

  public FullyConnectedLayer(int inputSize, int outputSize) {
    // Initialize weights and bias
    this.outputSize = outputSize;
    this.weights = Matrix.random(inputSize, outputSize); // Random weights
    this.bias = Matrix.random(1, outputSize); // Random bias
  }

  @Override
  public Matrix forward(Matrix input) {
    this.input = input;

    if (this.weights.getRows() != input.getCols()) {
      throw new IllegalArgumentException("Matrix inner dimensions must agree.");
    }
    Matrix dotResult = input.dot(this.weights);
    this.output = dotResult.add(this.bias);

    return this.output;
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    // Calculate gradient with respect to weights
    Matrix weightsGradient = this.input.transpose().dot(outputGradient);

    // Calculate gradient with respect to input
    Matrix inputGradient = outputGradient.dot(this.weights.transpose());

    // Update weights and bias
    this.weights = this.weights.subtract(weightsGradient.multiply(learningRate));
    this.bias = this.bias.subtract(outputGradient.multiply(learningRate));

    return inputGradient;
  }
}
