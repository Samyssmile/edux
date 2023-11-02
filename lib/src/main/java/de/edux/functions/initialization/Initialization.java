package de.edux.functions.initialization;

/**
 * Enumerates strategies for initializing weights in neural network layers, providing methods to
 * apply these strategies to given weight arrays.
 */
public enum Initialization {
  /**
   * Enumerates strategies for initializing weights in neural network layers, providing methods to
   * apply these strategies to given weight arrays.
   */
  XAVIER {
    /**
     * Applies Xavier initialization to the provided weights array using the input size.
     *
     * @param inputSize the size of the layer input
     * @param weights the array of weights to be initialized
     * @return the initialized weights array
     */
    @Override
    public double[] weightInitialization(int inputSize, double[] weights) {
      double xavier = Math.sqrt(6.0 / (inputSize + 1));
      for (int i = 0; i < weights.length; i++) {
        weights[i] = Math.random() * 2 * xavier - xavier;
      }
      return weights;
    }
  },

  /**
   * He initialization strategy for weights. This strategy is designed for layers with ReLU
   * activation, initializing the weights with variance scaled by the size of the previous layer,
   * aiming to reduce the vanishing gradient problem.
   */
  HE {
    /**
     * Applies He initialization to the provided weights array using the input size.
     *
     * @param inputSize the size of the layer input
     * @param weights the array of weights to be initialized
     * @return the initialized weights array
     */
    @Override
    public double[] weightInitialization(int inputSize, double[] weights) {
      double he = Math.sqrt(2.0 / inputSize);
      for (int i = 0; i < weights.length; i++) {
        weights[i] = Math.random() * 2 * he - he;
      }
      return weights;
    }
  };

  public abstract double[] weightInitialization(int inputSize, double[] weights);
}
