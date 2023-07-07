package de.edux.nn.network.api;

public interface IPerceptron {
    void train(double[][] inputs, double[][] targetOutputs);

    double[] predict(double[] inputs);

    void backpropagate(double[] inputs, double target);

    double evaluate(double[][] inputs, double[][] targetOutputs);
}
