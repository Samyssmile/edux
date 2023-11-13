package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.math.Matrix3D;

public class CrossEntropyLossV2 {

  public double computeLoss(Matrix3D predictions, Matrix3D labels) {

    // Überprüfen, ob die Dimensionen übereinstimmen
    if (predictions.getDepth() != labels.getDepth()
        || predictions.getRows() != labels.getRows()
        || predictions.getCols() != labels.getCols()) {
      throw new IllegalArgumentException(
          "Die Dimensionen von Vorhersagen und Labels müssen übereinstimmen.");
    }

    double loss = 0.0;
    for (int d = 0; d < predictions.getDepth(); d++) {
      for (int i = 0; i < predictions.getRows(); i++) {
        for (int j = 0; j < predictions.getCols(); j++) {
          loss -= labels.get(d, i, j) * Math.log(predictions.get(d, i, j));
        }
      }
    }
    return loss;
  }

  public Matrix3D computeGradient(Matrix3D predictions, Matrix3D labels) {
    // Überprüfen, ob die Dimensionen übereinstimmen
    if (predictions.getDepth() != labels.getDepth()
        || predictions.getRows() != labels.getRows()
        || predictions.getCols() != labels.getCols()) {
      throw new IllegalArgumentException(
          "Die Dimensionen von Vorhersagen und Labels müssen übereinstimmen.");
    }

    Matrix3D gradient =
        new Matrix3D(predictions.getDepth(), predictions.getRows(), predictions.getCols());
    for (int d = 0; d < predictions.getDepth(); d++) {
      for (int i = 0; i < predictions.getRows(); i++) {
        for (int j = 0; j < predictions.getCols(); j++) {
          double prediction = predictions.get(d, i, j);
          double label = labels.get(d, i, j);
          gradient.set(d, i, j, prediction - label);
        }
      }
    }
    return gradient;
  }
}
