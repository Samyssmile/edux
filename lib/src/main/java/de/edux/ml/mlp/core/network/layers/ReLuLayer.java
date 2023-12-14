package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;

public class ReLuLayer implements Layer {
    private Matrix lastInput;


    @Override
    public Matrix forwardLayerbased(Matrix input) {
        this.lastInput = input;
        return input.relu();
    }

    @Override
    public void updateWeightsAndBias() {
        //no weights and bias
    }

    @Override
    public Matrix backwardLayerBased(Matrix error, float learningRate) {
        return error.reluDerivative(lastInput);
    }


    @Override
    public String toString() {
        return "ReLu";
    }
}
