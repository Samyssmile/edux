package de.edux.functions.activation;

/**
 * Enumerates common activation functions used in neural networks and similar machine learning
 * architectures.
 *
 * <p>Each member of this enum represents a distinct type of activation function, a critical
 * component in neural networks. Activation functions determine the output of a neural network layer
 * for a given set of input, and they help normalize the output of each neuron to a specific range,
 * usually between 1 and -1 or between 1 and 0.
 *
 * <p>This enum simplifies the process of selecting and utilizing an activation function. It
 * provides an abstraction where the user can easily switch between different functions, making it
 * easier to experiment with neural network design. Additionally, each function includes a method
 * for calculating its derivative, which is essential for backpropagation in neural network
 * training.
 *
 * <p>Available functions include:
 *
 * <ul>
 *   <li><b>SIGMOID</b>: Normalizes inputs between 0 and 1, crucial for binary classification.
 *   <li><b>RELU</b>: Addresses the vanishing gradient problem, allowing for faster and more
 *       effective training.
 *   <li><b>LEAKY_RELU</b>: Variation of RELU, prevents "dying neurons" by allowing a small gradient
 *       when the unit is not active.
 *   <li><b>TANH</b>: Normalizes inputs between -1 and 1, a scaled version of the sigmoid function.
 *   <li><b>SOFTMAX</b>: Converts a vector of raw scores to a probability distribution, typically
 *       used in multi-class classification.
 * </ul>
 *
 * <p>Each function overrides the {@code calculateActivation} and {@code calculateDerivative}
 * methods, providing the specific implementation for the activation and its derivative based on
 * input. These are essential for the forward and backward passes through the network, respectively.
 *
 * <p><b>Note:</b> The {@code SOFTMAX} function additionally overrides {@code calculateActivation}
 * for an array input, facilitating its common use in output layers of neural networks for
 * classification tasks.
 */
public enum ActivationFunction {
  SIGMOID {
    @Override
    public double calculateActivation(double x) {
      return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double calculateDerivative(double x) {
      return calculateActivation(x) * (1 - calculateActivation(x));
    }
  },
  RELU {
    @Override
    public double calculateActivation(double x) {
      return Math.max(0, x);
    }

    @Override
    public double calculateDerivative(double x) {
      return x > 0 ? 1 : 0;
    }
  },
  LEAKY_RELU {
    @Override
    public double calculateActivation(double x) {
      return Math.max(0.01 * x, x);
    }

    @Override
    public double calculateDerivative(double x) {
      if (x > 0) {
        return 1.0;
      } else {
        return 0.01;
      }
    }
  },
  TANH {
    @Override
    public double calculateActivation(double x) {
      return Math.tanh(x);
    }

    @Override
    public double calculateDerivative(double x) {
      return 1 - Math.pow(calculateActivation(x), 2);
    }
  },
  SOFTMAX {
    @Override
    public double calculateActivation(double x) {
      return Math.exp(x);
    }

    @Override
    public double calculateDerivative(double x) {
      return calculateActivation(x) * (1 - calculateActivation(x));
    }

    @Override
    public double[] calculateActivation(double[] x) {
      double max = Double.NEGATIVE_INFINITY;
      for (double value : x) if (value > max) max = value;

      double sum = 0.0;
      for (int i = 0; i < x.length; i++) {
        x[i] = Math.exp(x[i] - max);
        sum += x[i];
      }

      for (int i = 0; i < x.length; i++) x[i] /= sum;

      return x;
    }
  };

  public abstract double calculateActivation(double x);

  public abstract double calculateDerivative(double x);

  public double[] calculateActivation(double[] x) {
    throw new UnsupportedOperationException("Not implemented");
  }
}
