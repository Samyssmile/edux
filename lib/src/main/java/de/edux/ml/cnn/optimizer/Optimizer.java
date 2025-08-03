package de.edux.ml.cnn.optimizer;

import de.edux.ml.cnn.tensor.Tensor;

import java.util.Map;

public interface Optimizer {
    void update(Map<Parameter, Tensor> params, Map<Parameter, Tensor> grads);
    void setLearningRate(double learningRate);
    double getLearningRate();
}