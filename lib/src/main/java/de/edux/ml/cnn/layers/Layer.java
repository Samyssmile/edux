package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix;

public interface Layer {
  Matrix forward(Matrix input);

  Matrix backward(Matrix outputGradient, double learningRate);
}
