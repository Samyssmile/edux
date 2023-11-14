package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.math.Matrix3D;

import de.edux.ml.cnn.math.Matrix3D;

public class CrossEntropyLossV2 {

  /**
   * Computes the cross-entropy loss.
   *
   * @param predicted The predicted output from the network.
   * @param actual The actual labels of the data.
   * @return The cross-entropy loss.
   */
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
          loss -= y * Math.log(p + 1e-15); // Adding a small constant to avoid log(0)
        }
      }
    }

    return loss / depth; // Average loss per item in the batch
  }

  /**
   * Computes the gradient of the cross-entropy loss.
   *
   * @param predicted The predicted output from the network.
   * @param actual The actual labels of the data.
   * @return The gradient of the loss.
   */
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
          gradient.set(d, r, c, p - y);
        }
      }
    }

    return gradient;
  }
}
