package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public class DropoutLayer implements Layer {
  private double dropoutRate;
  private Matrix mask;
  private boolean isTraining;

  public DropoutLayer(double dropoutRate) {
    this.dropoutRate = dropoutRate;
    this.isTraining = true; // Set to false during testing
  }

  @Override
  public Matrix forward(Matrix input) {
    if (!isTraining) {
      // During testing, we don't drop any units, but scale the output
      return input.multiply(1 - dropoutRate);
    }

    // Create a mask that randomly sets some elements to 0
    mask = Matrix.randomBinary(input.getRows(), input.getCols(), 1 - dropoutRate);
    return input.multiplyElementWise(mask);
  }

  @Override
  public Matrix backward(Matrix outputGradient, double learningRate) {
    // Only backpropagate through non-dropped out elements
    return outputGradient.multiplyElementWise(mask);
  }

  public void setTraining(boolean isTraining) {
    this.isTraining = isTraining;
  }
}
