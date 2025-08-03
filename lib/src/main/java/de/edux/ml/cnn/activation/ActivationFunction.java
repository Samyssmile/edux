package de.edux.ml.cnn.activation;

import de.edux.ml.cnn.tensor.Tensor;
import java.io.Serializable;

public interface ActivationFunction extends Serializable {
    Tensor apply(Tensor x);
    Tensor gradient(Tensor x);
}