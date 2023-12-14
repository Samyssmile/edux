package de.edux.ml.mlp.core.network;

import de.edux.ml.mlp.core.tensor.Matrix;
import java.io.Serializable;

public interface Layer extends Serializable {

    Matrix backwardLayerBased(Matrix error, float learningRate);

    Matrix forwardLayerbased(Matrix input);

    void updateWeightsAndBias();
}
