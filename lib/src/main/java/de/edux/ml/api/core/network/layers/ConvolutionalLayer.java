package de.edux.ml.api.core.network.layers;

import de.edux.ml.api.core.network.Layer;
import de.edux.ml.api.core.tensor.Matrix;

public class ConvolutionalLayer implements Layer {
  @Override
  public Matrix backwardLayerBased(Matrix error, float learningRate) {
    return null;
  }

  @Override
  public Matrix forwardLayerbased(Matrix input) {
    return null;
  }

  @Override
  public void updateWeightsAndBias() {}
}
