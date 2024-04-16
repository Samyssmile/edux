package de.edux.ml.api.core.network.loss;

import de.edux.ml.api.core.tensor.Matrix;

public class LossFunctions {

  public static Matrix crossEntropy(Matrix expected, Matrix actual) {
    return actual
        .apply((index, value) -> -expected.getData()[index] * Math.log(value))
        .sumColumns();
  }
}
