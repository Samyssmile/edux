package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.math.Matrix;

public class CrossEntropyLoss {

  /**
   * Berechnet den Cross-Entropy-Verlust.
   *
   * @param predicted Die vorhergesagten Wahrscheinlichkeiten, normalerweise die Ausgabe des
   *     Netzwerks.
   * @param actual Die tatsächlichen Klassen (normalerweise in One-Hot-Encoding).
   * @return Den Cross-Entropy-Verlust.
   */
  public static double computeLoss(Matrix predicted, Matrix actual) {
    int m = actual.getRows();
    double sum = 0.0;

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < actual.getCols(); j++) {
        double p = predicted.getData()[i][j];
        double y = actual.getData()[i][j];
        sum += y * Math.log(p);
      }
    }

    return -sum / m;
  }

  /**
   * Berechnet den Gradienten des Cross-Entropy-Verlusts.
   *
   * @param predicted Die vorhergesagten Wahrscheinlichkeiten, normalerweise die Ausgabe des
   *     Netzwerks.
   * @param actual Die tatsächlichen Klassen (normalerweise in One-Hot-Encoding).
   * @return Den Gradienten des Verlustes.
   */
  public static Matrix computeGradient(Matrix predicted, Matrix actual) {
    return predicted.subtract(actual).multiply(1.0 / actual.getRows());
  }
}
