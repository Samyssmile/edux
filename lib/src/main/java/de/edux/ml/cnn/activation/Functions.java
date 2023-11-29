package de.edux.ml.cnn.activation;

public class Functions {
  public static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

    public static double sigmoidDerivative(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

}
