package de.edux.ml.cnn.layers;

import de.edux.ml.cnn.math.Matrix3D;

public interface Layer {
  Matrix3D forward(Matrix3D input);

  Matrix3D backward(Matrix3D outputGradient, double learningRate);
}
