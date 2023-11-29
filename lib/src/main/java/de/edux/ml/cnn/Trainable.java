package de.edux.ml.cnn;

import de.edux.ml.cnn.core.Tensor;

public interface Trainable {

    public void train(Tensor inputs, Tensor targets, int epochs, int batchSize, double learningRate);

    public Tensor predict(Tensor input);

    public double evaluate(Tensor inputs, Tensor targets);
}
