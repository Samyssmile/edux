package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.Tensor;

public interface Layer {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput);
    void setTraining(boolean training);
    boolean isTraining();
    
    default void zeroGradients() {
    }
    
    default void cleanup() {
    }
}