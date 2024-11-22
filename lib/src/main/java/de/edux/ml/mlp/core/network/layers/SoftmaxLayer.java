package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;

public class SoftmaxLayer implements Layer {

    private Matrix lastSoftmax;

    @Override
    public Matrix backwardLayerBased(Matrix expected, float learningRate) {
        /*Check that output cotains NaN*/
        for (int i = 0; i < expected.getRows(); i++) {
            for (int j = 0; j < expected.getCols(); j++) {
                if (Double.isNaN(expected.get(i, j))) {
                    throw new RuntimeException("NaN in output");
                }
            }
        }
        Matrix output = lastSoftmax.subtract(expected);

        for (int i = 0; i < output.getRows(); i++) {
            for (int j = 0; j < output.getCols(); j++) {
                if (Double.isNaN(output.get(i, j))) {
                    throw new RuntimeException("NaN in output");
                }
            }
        }
        return output;
    }

    @Override
    public Matrix forwardLayerbased(Matrix input) {
        this.lastSoftmax = input.softmax();
        if (lastSoftmax.hasNaN()) {
            throw new RuntimeException("NaN in output");
        }
        return lastSoftmax;
    }

    @Override
    public void updateWeightsAndBias() {
        //no weights and bias
    }

    @Override
    public String toString() {
        return "Softmax";
    }
}
