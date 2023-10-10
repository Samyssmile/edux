package de.edux.ml.nn.network;

import de.edux.api.Classifier;
import de.edux.functions.activation.ActivationFunction;
import de.edux.ml.nn.Neuron;
import de.edux.ml.nn.config.NetworkConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * The {@code MultilayerPerceptron} class represents a simple feedforward neural network,
 * which consists of input, hidden, and output layers. It implements the {@code Classifier}
 * interface, facilitating both the training and prediction processes on a given dataset.
 *
 * <p>This implementation utilizes a backpropagation algorithm for training the neural network
 * to adjust weights and biases, considering a set configuration defined by {@link NetworkConfiguration}.
 * The network's architecture is multi-layered, comprising one or more hidden layers in addition
 * to the input and output layers. Neurons within these layers utilize activation functions defined
 * per layer through the configuration.</p>
 *
 * <p>The training process adjusts the weights and biases of neurons within the network based on
 * the error between predicted and expected outputs. Additionally, the implementation provides functionality
 * to save and restore the best model achieved during training based on accuracy. Early stopping is applied
 * during training to prevent overfitting and unnecessary computational expense by monitoring the performance
 * improvement across epochs.</p>
 *
 * <p>Usage example:</p>
 * <pre>
 *    NetworkConfiguration config = ... ;
 *    double[][] testFeatures = ... ;
 *    double[][] testLabels = ... ;
 *
 *    MultilayerPerceptron mlp = new MultilayerPerceptron(config, testFeatures, testLabels);
 *    mlp.train(features, labels);
 *
 *    double accuracy = mlp.evaluate(testFeatures, testLabels);
 *    double[] prediction = mlp.predict(singleInput);
 * </pre>
 *
 * <p>Note: This implementation logs informative messages, such as accuracy per epoch, using SLF4J logging.</p>
 *
 * @see de.edux.api.Classifier
 * @see de.edux.ml.nn.Neuron
 * @see de.edux.ml.nn.config.NetworkConfiguration
 * @see de.edux.functions.activation.ActivationFunction
 */
public class MultilayerPerceptron implements Classifier {
    private static final Logger LOG = LoggerFactory.getLogger(MultilayerPerceptron.class);

    private final NetworkConfiguration config;
    private final ActivationFunction hiddenLayerActivationFunction;
    private final ActivationFunction outputLayerActivationFunction;
    private List<Neuron[]> hiddenLayers;
    private Neuron[] outputLayer;
    private final double[][] testFeatures;
    private final double[][] testLabels;
    private double bestAccuracy;
    private ArrayList<Neuron[]> bestHiddenLayers;
    private Neuron[] bestOutputLayer;

    public MultilayerPerceptron(NetworkConfiguration config, double[][] testFeatures, double[][] testLabels) {
        this.config = config;
        this.testFeatures = testFeatures;
        this.testLabels = testLabels;

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

        for (Neuron[] layer : hiddenLayers) {
            double[] hiddenOutputs = new double[layer.length];
            for (int i = 0; i < layer.length; i++) {
                hiddenOutputs[i] = layer[i].calculateOutput(currentInput);
            }
            currentInput = hiddenOutputs;
        }

        double[] output = new double[config.outputSize()];
        for (int i = 0; i < config.outputSize(); i++) {
            output[i] = outputLayer[i].calculateOutput(currentInput);
        }

        return outputLayerActivationFunction.calculateActivation(output);
    }

    @Override
    public boolean train(double[][] features, double[][] labels) {
        bestAccuracy = 0;
        int epochsWithoutImprovement = 0;
        final int PATIENCE = 10;

        for (int epoch = 0; epoch < config.epochs(); epoch++) {
            for (int i = 0; i < features.length; i++) {
                double[] output = feedforward(features[i]);

                double[] output_error_signal = new double[config.outputSize()];
                for (int j = 0; j < config.outputSize(); j++) {
                    output_error_signal[j] = labels[i][j] - output[j];
                }

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


                updateWeights(i, output_error_signal, hidden_error_signals, features);
            }

            double accuracy = evaluate(testFeatures, testLabels);
            LOG.info("Epoch: {} - Accuracy: {}%", epoch, String.format("%.2f", accuracy * 100));

            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                epochsWithoutImprovement = 0;
                saveBestModel(hiddenLayers, outputLayer);
            } else {
                epochsWithoutImprovement++;
            }

            if (epochsWithoutImprovement >= PATIENCE) {
                LOG.info("Early stopping: Stopping training as the model has not improved in the last {} epochs.", PATIENCE);
                loadBestModel();
                LOG.info("Best accuracy after restoring best MLP model: {}%", String.format("%.2f", bestAccuracy * 100));
                break;
            }
        }
        return true;
    }

    private void loadBestModel() {
        this.hiddenLayers = this.bestHiddenLayers;
        this.outputLayer = this.bestOutputLayer;
    }

    private void saveBestModel(List<Neuron[]> hiddenLayers, Neuron[] outputLayer) {
        this.bestHiddenLayers = new ArrayList<>();
        this.bestOutputLayer = new Neuron[outputLayer.length];
        for (int i = 0; i < hiddenLayers.size(); i++) {
            Neuron[] layer = hiddenLayers.get(i);
            Neuron[] newLayer = new Neuron[layer.length];
            for (int j = 0; j < layer.length; j++) {
                newLayer[j] = new Neuron(layer[j].getWeights().length, layer[j].getActivationFunction(), layer[j].getInitialization());
                newLayer[j].setBias(layer[j].getBias());
                for (int k = 0; k < layer[j].getWeights().length; k++) {
                    newLayer[j].getWeights()[k] = layer[j].getWeight(k);
                }
            }
            this.bestHiddenLayers.add(newLayer);
        }
        for (int i = 0; i < outputLayer.length; i++) {
            this.bestOutputLayer[i] = new Neuron(outputLayer[i].getWeights().length, outputLayer[i].getActivationFunction(), outputLayer[i].getInitialization());
            this.bestOutputLayer[i].setBias(outputLayer[i].getBias());
            for (int j = 0; j < outputLayer[i].getWeights().length; j++) {
                this.bestOutputLayer[i].getWeights()[j] = outputLayer[i].getWeight(j);
            }
        }

    }

    private void updateWeights(int i, double[] output_error_signal, List<double[]> hidden_error_signals, double[][] features) {
        double[] currentInput = features[i];

        for (int j = 0; j < hiddenLayers.size(); j++) {
            Neuron[] layer = hiddenLayers.get(j);
            double[] errorSignal = hidden_error_signals.get(j);
            for (int k = 0; k < layer.length; k++) {
                layer[k].adjustBias(errorSignal[k], config.learningRate());
                layer[k].adjustWeights(currentInput, errorSignal[k], config.learningRate());
            }
            currentInput = new double[layer.length];
            for (int k = 0; k < layer.length; k++) {
                currentInput[k] = layer[k].calculateOutput(features[i]);
            }
        }

        for (int j = 0; j < config.outputSize(); j++) {
            outputLayer[j].adjustBias(output_error_signal[j], config.learningRate());
            outputLayer[j].adjustWeights(currentInput, output_error_signal[j], config.learningRate());
        }
    }

    @Override
    public double evaluate(double[][] testInputs, double[][] testTargets) {
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

}
