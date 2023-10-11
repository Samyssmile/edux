package de.edux.edux.activation;

import de.edux.functions.activation.ActivationFunction;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ActivationFunctionTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testSigmoid() {
        double input = 0.5;
        double expectedActivation = 1 / (1 + Math.exp(-input));
        double expectedDerivative = expectedActivation * (1 - expectedActivation);

        Assertions.assertEquals(expectedActivation, ActivationFunction.SIGMOID.calculateActivation(input), DELTA);
        assertEquals(expectedDerivative, ActivationFunction.SIGMOID.calculateDerivative(input), DELTA);
    }

    @Test
    public void testRelu() {
        double inputPositive = 0.5;
        double inputNegative = -0.5;

        assertEquals(inputPositive, ActivationFunction.RELU.calculateActivation(inputPositive), DELTA);
        assertEquals(0.0, ActivationFunction.RELU.calculateActivation(inputNegative), DELTA);

        assertEquals(1.0, ActivationFunction.RELU.calculateDerivative(inputPositive), DELTA);
        assertEquals(0.0, ActivationFunction.RELU.calculateDerivative(inputNegative), DELTA);
    }

    @Test
    public void testLeakyRelu() {
        double inputPositive = 0.5;
        double inputNegative = -0.5;

        assertEquals(inputPositive, ActivationFunction.LEAKY_RELU.calculateActivation(inputPositive), DELTA);
        assertEquals(0.01 * inputNegative, ActivationFunction.LEAKY_RELU.calculateActivation(inputNegative), DELTA);

        assertEquals(1.0, ActivationFunction.LEAKY_RELU.calculateDerivative(inputPositive), DELTA);
        assertEquals(0.01, ActivationFunction.LEAKY_RELU.calculateDerivative(inputNegative), DELTA);
    }

    @Test
    public void testTanh() {
        double input = 0.5;
        double expectedActivation = Math.tanh(input);
        double expectedDerivative = 1 - Math.pow(expectedActivation, 2);

        assertEquals(expectedActivation, ActivationFunction.TANH.calculateActivation(input), DELTA);
        assertEquals(expectedDerivative, ActivationFunction.TANH.calculateDerivative(input), DELTA);
    }

    @Test
    public void testSoftmax() {
        double input = 0.5;
        double expectedActivation = Math.exp(input);
        double expectedDerivative = expectedActivation * (1 - expectedActivation);

        assertEquals(expectedActivation, ActivationFunction.SOFTMAX.calculateActivation(input), DELTA);
        assertEquals(expectedDerivative, ActivationFunction.SOFTMAX.calculateDerivative(input), DELTA);

        double[] inputs = {0.1, 0.2, 0.3};
        double[] expectedOutputs = new double[inputs.length];
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            expectedOutputs[i] = Math.exp(inputs[i]);
            sum += expectedOutputs[i];
        }
        for (int i = 0; i < expectedOutputs.length; i++) {
            expectedOutputs[i] /= sum;
        }

        double[] outputs = ActivationFunction.SOFTMAX.calculateActivation(inputs);

        for (int i = 0; i < inputs.length; i++) {
            assertEquals(expectedOutputs[i], outputs[i], DELTA);
        }
    }
}
