package de.edux.ml.api.core.network.layers;

import de.edux.ml.api.core.network.Layer;
import de.edux.ml.api.core.tensor.Matrix;

public class SoftmaxLayer implements Layer {

  private Matrix lastSoftmax;

  @Override
  public Matrix backwardLayerBased(Matrix expected, float learningRate) {
    return lastSoftmax.subtract(expected);
  }

  @Override
  public Matrix forwardLayerbased(Matrix input) {
    this.lastSoftmax = input.softmax();
    ;
    return lastSoftmax;
  }

  @Override
  public void updateWeightsAndBias() {
    // no weights and bias
  }

  @Override
  public String toString() {
    return "Softmax";
  }
}
