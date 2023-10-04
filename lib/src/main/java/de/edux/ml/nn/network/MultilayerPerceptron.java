package de.edux.ml.nn.network;

import de.edux.functions.initialization.Initialization;
import de.edux.ml.nn.Neuron;
import de.edux.functions.activation.ActivationFunction;
import de.edux.ml.nn.config.NetworkConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class MultilayerPerceptron {
    private static final Logger LOG = LoggerFactory.getLogger(MultilayerPerceptron.class);

    private final double[][] inputs;
    private final double[][] targets;
    private final NetworkConfiguration config;
    private final ActivationFunction hiddenLayerActivationFunction;
    private final ActivationFunction outputLayerActivationFunction;
    private final double[][] testInputs;
    private final double[][] testTargets;
    private final List<Neuron[]> hiddenLayers;
    private final Neuron[] outputLayer;
    private double bestAccuracy;

    public MultilayerPerceptron(double[][] inputs, double[][] targets, double[][] testInputs, double[][] testTargets, NetworkConfiguration config) {
        this.inputs = inputs;
        this.targets = targets;
        this.testInputs = testInputs;
        this.testTargets = testTargets;
        this.config = config;

        hiddenLayerActivationFunction = config.hiddenLayerActivationFunction();
        outputLayerActivationFunction = config.outputLayerActivationFunction();

        hiddenLayers = new ArrayList<>();

        int inputSizeForCurrentLayer = config.inputSize();
        for (int layerSize : config.hiddenLayersSize()) {
            Neuron[] hiddenLayer = new Neuron[layerSize];
            for (int i = 0; i < layerSize; i++) {
                hiddenLayer[i] = new Neuron(inputSizeForCurrentLayer, hiddenLayerActivationFunction, this.config.hiddenLayerWeightInitialization());
            }
            hiddenLayers.add(hiddenLayer);
            inputSizeForCurrentLayer = layerSize;
        }

        outputLayer = new Neuron[config.outputSize()];
        for (int i = 0; i < config.outputSize(); i++) {
            outputLayer[i] = new Neuron(inputSizeForCurrentLayer, outputLayerActivationFunction, this.config.outputLayerWeightInitialization());
        }
    }

    private double[] feedforward(double[] input) {
        double[] currentInput = input;

        // Pass input through all hidden layers
        for (Neuron[] layer : hiddenLayers) {
            double[] hiddenOutputs = new double[layer.length];
            for (int i = 0; i < layer.length; i++) {
                hiddenOutputs[i] = layer[i].calculateOutput(currentInput);
            }
            currentInput = hiddenOutputs;
        }

        // Pass input through output layer
        double[] output = new double[config.outputSize()];
        for (int i = 0; i < config.outputSize(); i++) {
            output[i] = outputLayer[i].calculateOutput(currentInput);
        }

        return outputLayerActivationFunction.calculateActivation(output);
    }

    public void train() {
         bestAccuracy = 0;
        for (int epoch = 0; epoch < config.epochs(); epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] output = feedforward(inputs[i]);

                // Calculate error signals
                double[] output_error_signal = new double[config.outputSize()];
                for (int j = 0; j < config.outputSize(); j++)
                    output_error_signal[j] = targets[i][j] - output[j];

                List<double[]> hidden_error_signals = new ArrayList<>();
                for (int j = hiddenLayers.size() - 1; j >= 0; j--) {
                    double[] hidden_error_signal = new double[hiddenLayers.get(j).length];
                    for (int k = 0; k < hiddenLayers.get(j).length; k++) {
                        for (int l = 0; l < output_error_signal.length; l++) {
                            hidden_error_signal[k] += output_error_signal[l] * (j == hiddenLayers.size() - 1 ? outputLayer[l].getWeight(k) : hiddenLayers.get(j + 1)[l].getWeight(k));
                        }
                    }
                    hidden_error_signals.add(0, hidden_error_signal);
                    output_error_signal = hidden_error_signal;
                }

                updateWeights(i, output_error_signal, hidden_error_signals);
            }

            if (epoch % 10 == 0) {
                double accuracy = evaluate(testInputs, testTargets) * 100;
                if (accuracy == 100) {
                    LOG.info("Stop training at: {}%", String.format("%.2f", accuracy));
                    return;
                }

                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    LOG.info("Best Accuracy: {}%", String.format("%.2f", bestAccuracy));
                }
                // if accuracy 20% lower than best accuracy, stop training
                if (bestAccuracy - accuracy > 20) {
                    LOG.info("Local Minama found, stop training");
                    return;
                }
            }
        }
    }

    private void updateWeights(int i, double[] output_error_signal, List<double[]> hidden_error_signals) {
        double[] currentInput = inputs[i];

        for (int j = 0; j < hiddenLayers.size(); j++) {
            Neuron[] layer = hiddenLayers.get(j);
            double[] errorSignal = hidden_error_signals.get(j);
            for (int k = 0; k < layer.length; k++) {
                layer[k].adjustBias(errorSignal[k], config.learningRate());
                layer[k].adjustWeights(currentInput, errorSignal[k], config.learningRate());
            }
            currentInput = new double[layer.length];
            for (int k = 0; k < layer.length; k++) {
                currentInput[k] = layer[k].calculateOutput(inputs[i]);
            }
        }

        for (int j = 0; j < config.outputSize(); j++) {
            outputLayer[j].adjustBias(output_error_signal[j], config.learningRate());
            outputLayer[j].adjustWeights(currentInput, output_error_signal[j], config.learningRate());
        }
    }

    private double evaluate(double[][] testInputs, double[][] testTargets) {
        int correctCount = 0;

        for (int i = 0; i < testInputs.length; i++) {
            double[] predicted = predict(testInputs[i]);
            int predictedIndex = 0;
            int targetIndex = 0;

            for (int j = 0; j < predicted.length; j++) {
                if (predicted[j] > predicted[predictedIndex])
                    predictedIndex = j;
                if (testTargets[i][j] > testTargets[i][targetIndex])
                    targetIndex = j;
            }

            if (predictedIndex == targetIndex)
                correctCount++;
        }

        return (double) correctCount / testInputs.length;
    }

    public double[] predict(double[] input) {
        return feedforward(input);
    }

    public double getAccuracy() {
        return bestAccuracy;
    }
}
