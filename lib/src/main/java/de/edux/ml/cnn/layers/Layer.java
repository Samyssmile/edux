package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.core.Tensor;

public abstract class Layer {

    public Tensor weights;
    public Tensor biases;

    public abstract Tensor forward(Tensor input);

    public abstract Tensor backward(Tensor input, double learningRate);

}
