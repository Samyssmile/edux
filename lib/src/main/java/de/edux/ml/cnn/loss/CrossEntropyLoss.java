package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.math.Matrix;

public class CrossEntropyLoss {

  public double computeLoss(Matrix predicted, Matrix labels) {
    int m = labels.getRows();
    double sum = 0.0;

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < labels.getCols(); j++) {
        double p = predicted.getData()[i][j];
        double y = labels.getData()[i][j];
        sum += y * Math.log(p);
      }
    }

    return -sum / m;
  }

  public Matrix computeGradient(Matrix predicted, Matrix labels) {
    return predicted.subtract(labels).multiply(1.0 / labels.getRows());
  }
}
