package de.edux.ml.nn.network;

import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;

class Neuron {
  private final Initialization initialization;
  private final ActivationFunction activationFunction;
  private double[] weights;
  private double bias;

  public Neuron(
      int inputSize, ActivationFunction activationFunction, Initialization initialization) {
    this.weights = new double[inputSize];
    this.activationFunction = activationFunction;
    this.initialization = initialization;
    this.bias = initialization.weightInitialization(inputSize, new double[1])[0];
    this.weights = initialization.weightInitialization(inputSize, weights);
  }

  public Initialization getInitialization() {
    return initialization;
  }

  public double calculateOutput(double[] input) {
    double output = bias;
    for (int i = 0; i < input.length; i++) {
      output += input[i] * weights[i];
    }
    return activationFunction.calculateActivation(output);
  }

  public void adjustWeights(double[] input, double error, double learningRate) {
    for (int i = 0; i < weights.length; i++) {
      weights[i] += learningRate * input[i] * error;
    }
  }

  public void adjustBias(double error, double learningRate) {
    bias += learningRate * error;
  }

  public double getWeight(int index) {
    return weights[index];
  }

  public double[] getWeights() {
    return weights;
  }

  public void setWeights(double[] weights) {
    this.weights = weights;
  }

  public double getBias() {
    return bias;
  }

  public void setBias(double bias) {
    this.bias = bias;
  }

  public ActivationFunction getActivationFunction() {
    return activationFunction;
  }
}
