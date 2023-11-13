package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public interface IMatrix {

  // Grundlegende Getter-Methoden
  double[][][] getData();

  int getRows();

  int getCols();
}
