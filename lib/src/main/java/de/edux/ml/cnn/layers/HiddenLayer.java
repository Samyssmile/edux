package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.core.Tensor;
import java.util.function.Function;

public class HiddenLayer extends Layer {

  private int inputSize;
  private int outputSize;
  private Function<Double, Double> activationFunction;
  private Function<Double, Double> activationDerivative;

  public HiddenLayer(int inputSize, int outputSize) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = new Tensor(outputSize, inputSize);
    this.biases = new Tensor(outputSize, 1);
    initializeWeights();

    // ReLU activation and its derivative
    this.activationFunction = (x) -> Math.max(0, x);
    this.activationDerivative = (x) -> (x > 0) ? 1.0 : 0.0;
  }

  private void initializeWeights() {
    // Initialize weights and biases (simple random initialization)
    for (int i = 0; i < weights.getData().length; i++) {
      for (int j = 0; j < weights.getData()[0].length; j++) {
        weights.getData()[i][j] = Math.random() - 0.5; // Random weights between -0.5 and 0.5
      }
      biases.getData()[i][0] = 0; // Initialize biases to 0
    }
  }

  @Override
  public Tensor forward(Tensor input) {
    Tensor output = new Tensor(outputSize, 1);
    for (int i = 0; i < outputSize; i++) {
      double sum = 0;
      for (int j = 0; j < inputSize; j++) {
        sum += weights.getData()[i][j] * input.getData()[j][0];
      }
      sum += biases.getData()[i][0];
      output.getData()[i][0] = activationFunction.apply(sum);
    }
    return output;
  }

  @Override
  public Tensor backward(Tensor input, double learningRate) {
    Tensor gradients = new Tensor(inputSize, 1);

    // Berechnen Sie den Gradienten für jedes Gewicht
    for (int i = 0; i < outputSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        double outputDerivative = activationDerivative.apply(weights.getData()[i][j]);
        gradients.getData()[j][0] += outputDerivative * input.getData()[j][0];
        weights.getData()[i][j] -= learningRate * outputDerivative * input.getData()[j][0];
      }
      biases.getData()[i][0] -= learningRate * activationDerivative.apply(biases.getData()[i][0]);
    }

    return gradients;
  }
}

