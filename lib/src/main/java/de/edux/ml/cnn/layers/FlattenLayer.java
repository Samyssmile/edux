package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public class FlattenLayer implements Layer {
  private int[] inputShape;

  @Override
  public Matrix forward(Matrix input) {
    // Store the shape of the input to use in the backward pass
    this.inputShape = new int[] {input.getRows(), input.getCols()};

    // Flatten the input
    return input.flatten();
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    return outputGradient.reshape(inputShape[0], inputShape[1]);
  }
}
