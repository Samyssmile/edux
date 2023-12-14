package de.edux.ml.mlp.core.network.loss;

import de.edux.ml.mlp.core.tensor.Matrix;

public class LossFunctions {

    public static Matrix crossEntropy(Matrix expected, Matrix actual){
        return  actual.apply((index, value) -> -expected.getData()[index] * Math.log(value)).sumColumns();
    }

}
