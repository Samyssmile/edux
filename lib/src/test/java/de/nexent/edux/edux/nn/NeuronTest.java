package de.nexent.edux.edux.nn;

import de.nexent.edux.functions.activation.ActivationFunction;
import de.nexent.edux.ml.nn.Neuron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NeuronTest {

    private Neuron neuron;
    private final int inputSize = 3;
    private final ActivationFunction dummyActivationFunction = ActivationFunction.SOFTMAX;

    @BeforeEach
    public void setUp() {
        neuron = new Neuron(inputSize, dummyActivationFunction);
    }

    @Test
    public void testAdjustWeights() {
        double[] initialWeights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            initialWeights[i] = neuron.getWeight(i);
        }

        double[] input = {1.0, 2.0, 3.0};
        double error = 0.5;
        double learningRate = 0.1;
        neuron.adjustWeights(input, error, learningRate);

        for (int i = 0; i < inputSize; i++) {
            double expectedWeight = initialWeights[i] + learningRate * input[i] * error;
            assertEquals(expectedWeight, neuron.getWeight(i));
        }
    }

    @Test
    public void testAdjustBias() {
        double initialBias = neuron.getBias();

        double error = 0.5;
        double learningRate = 0.1;
        neuron.adjustBias(error, learningRate);

        double expectedBias = initialBias + learningRate * error;
        assertEquals(expectedBias, neuron.getBias());
    }
}
