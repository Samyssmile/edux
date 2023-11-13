package de.edux.ml.cnn.math;

public interface IMatrix3D {

  Matrix3D convolve(Matrix3D kernel, int stride, int padding);

  Matrix3D maxPooling(int poolSize, int stride);

  Matrix3D applyReLU();

  Matrix3D applyLeakyReLU();

  Matrix3D applyPadding(int padding);

  Matrix3D flatten();

  Matrix3D convolveBackprop(Matrix3D gradient, int stride, int padding);

  Matrix3D add(Matrix3D other);

  Matrix3D subtract(Matrix3D other);

  Matrix3D multiplyElementWise(Matrix3D other);

  void normalize(double mean, double std);

  double get(int depth, int row, int col);

  void set(int depth, int row, int col, double value);

  int getDepth();

  int getRows();

  int getCols();
}
