package de.edux.ml.nn;

import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;

public class Neuron {
    private double[] weights;
    private double bias;
    private final ActivationFunction activationFunction;

    public Neuron(int inputSize, ActivationFunction activationFunction, Initialization initialization) {
        this.weights = new double[inputSize];
        this.activationFunction = activationFunction;
        this.bias = initialization.weightInitialization(inputSize, new double[1])[0];
        this.weights = initialization.weightInitialization(inputSize, weights);

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

    public double getBias() {
        return bias;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
