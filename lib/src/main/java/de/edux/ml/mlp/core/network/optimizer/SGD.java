package de.edux.ml.mlp.core.network.optimizer;

import de.edux.ml.mlp.core.tensor.Matrix;

public class SGD {
    public void updateWeights(Matrix weights, Matrix weightErrors, float learningRate) {
        for (int i = 0; i < weights.getRows(); i++) {
            for (int j = 0; j < weights.getCols(); j++) {
                weights.set(i, j, weights.get(i, j) - learningRate * weightErrors.get(i, j));
            }
        }
    }
}
