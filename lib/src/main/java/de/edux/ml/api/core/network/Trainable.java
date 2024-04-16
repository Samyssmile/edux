package de.edux.ml.api.core.network;

import de.edux.ml.api.core.tensor.Matrix;

public interface Trainable {

  public void train(Matrix input, Matrix expected, double learningRate, int epochs);
}
