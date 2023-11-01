package de.edux.functions.activation;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class ActivationFunctionTest {

  private static final double DELTA = 1e-6;

  @Test
  public void sigmoidActivation_andDerivative() {
    double x = 0.5;
    double expectedActivation = 1 / (1 + Math.exp(-x));
    double expectedDerivative = expectedActivation * (1 - expectedActivation);

    assertEquals(expectedActivation, ActivationFunction.SIGMOID.calculateActivation(x), DELTA);
    assertEquals(expectedDerivative, ActivationFunction.SIGMOID.calculateDerivative(x), DELTA);
  }

  @Test
  public void reluActivation_andDerivative() {
    double positiveX = 0.5;
    double negativeX = -0.5;

    assertEquals(positiveX, ActivationFunction.RELU.calculateActivation(positiveX), DELTA);
    assertEquals(1.0, ActivationFunction.RELU.calculateDerivative(positiveX), DELTA);

    assertEquals(0.0, ActivationFunction.RELU.calculateActivation(negativeX), DELTA);
    assertEquals(0.0, ActivationFunction.RELU.calculateDerivative(negativeX), DELTA);
  }

  @Test
  public void leakyReluActivation_andDerivative() {
    double positiveX = 0.5;
    double negativeX = -0.5;

    assertEquals(positiveX, ActivationFunction.LEAKY_RELU.calculateActivation(positiveX), DELTA);
    assertEquals(1.0, ActivationFunction.LEAKY_RELU.calculateDerivative(positiveX), DELTA);

    assertEquals(
        0.01 * negativeX, ActivationFunction.LEAKY_RELU.calculateActivation(negativeX), DELTA);
    assertEquals(0.01, ActivationFunction.LEAKY_RELU.calculateDerivative(negativeX), DELTA);
  }

  @Test
  public void tanhActivation_andDerivative() {
    double x = 0.5;
    double expectedActivation = Math.tanh(x);
    double expectedDerivative = 1 - Math.pow(expectedActivation, 2);

    assertEquals(expectedActivation, ActivationFunction.TANH.calculateActivation(x), DELTA);
    assertEquals(expectedDerivative, ActivationFunction.TANH.calculateDerivative(x), DELTA);
  }

  @Test
  public void softmaxActivation() {
    double[] x = {1.0, 2.0, 3.0};
    double[] softmax = ActivationFunction.SOFTMAX.calculateActivation(x.clone());

    double sum = 0.0;
    for (double value : softmax) {
      sum += value;
    }

    assertEquals(1.0, sum, DELTA, "The sum of the softmax probabilities should be 1.");
    assertTrue(
        softmax[0] < softmax[1] && softmax[1] < softmax[2],
        "Softmax values should reflect the order of inputs.");
  }
}
