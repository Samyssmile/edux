package de.edux.ml.mlp.core.network;

import de.edux.ml.mlp.core.tensor.Matrix;

public interface Trainable {

    public void train(Matrix input, Matrix expected, double learningRate, int epochs);
}
