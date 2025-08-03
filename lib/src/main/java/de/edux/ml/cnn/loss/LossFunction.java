package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.tensor.Tensor;
import java.io.Serializable;

public interface LossFunction {
    LossOutput compute(Tensor predictions, Tensor labels);
}