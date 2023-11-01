package de.edux.functions;

import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.functions.initialization.Initialization;
import org.junit.jupiter.api.Test;

public class InitializationTest {

  @Test
  public void testXavierInitialization() {
    int inputSize = 10;
    double[] weights = new double[inputSize];
    weights = Initialization.XAVIER.weightInitialization(inputSize, weights);

    double xavier = Math.sqrt(6.0 / (inputSize + 1));
    for (double weight : weights) {
      assertTrue(
          weight >= -xavier && weight <= xavier,
          "Weight should be in the range of Xavier initialization");
    }
  }

  @Test
  public void testHeInitialization() {
    int inputSize = 10;
    double[] weights = new double[inputSize];
    weights = Initialization.HE.weightInitialization(inputSize, weights);

    double he = Math.sqrt(2.0 / inputSize);
    for (double weight : weights) {
      assertTrue(
          weight >= -he && weight <= he, "Weight should be in the range of He initialization");
    }
  }
}
