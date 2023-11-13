package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.math.Matrix3D;

public class CrossEntropyLossV2 {

  public double computeLoss(Matrix3D predicted, Matrix3D actual) {
    int depth = predicted.getDepth();
    int rows = predicted.getRows();
    int cols = predicted.getCols();

    double loss = 0.0;
    for (int d = 0; d < depth; d++) {
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          double p = predicted.get(d, r, c);
          double y = actual.get(d, r, c);
          loss -=
              y * Math.log(p + 1e-15); // Vermeidet Log(0) durch Hinzufügen einer kleinen Konstante
        }
      }
    }
    return loss;
  }

  public Matrix3D computeGradient(Matrix3D predicted, Matrix3D actual) {
    int depth = predicted.getDepth();
    int rows = predicted.getRows();
    int cols = predicted.getCols();

    Matrix3D gradient = new Matrix3D(depth, rows, cols);

    for (int d = 0; d < depth; d++) {
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          double p = predicted.get(d, r, c);
          double y = actual.get(d, r, c);
          gradient.set(
              d, r, c,
              p - y); // Gradient ist die Differenz zwischen Vorhersage und tatsächlichem Wert
        }
      }
    }

    return gradient;
  }
}
