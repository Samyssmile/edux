package de.edux.ml.cnn.math;

/** public Matrix3D(int depth, int rows, int cols) */
public interface IMatrix3D {

  Matrix3D dot(Matrix3D other);

  Matrix3D convolve(Matrix3D kernel, int stride, int padding);

  Matrix3D maxPooling(int poolSize, int stride);

  Matrix3D applyReLU();

  Matrix3D applyReLUBackward(Matrix3D outputGradient);

  Matrix3D applyLeakyReLU();

  Matrix3D applyPadding(int padding);

  Matrix3D flatten();

  Matrix3D reshapeBack(int originalDepth, int originalRows, int originalCols);

  Matrix3D convolveBackprop(Matrix3D gradient, int stride, int padding);

  Matrix3D add(Matrix3D other);

  Matrix3D subtract(Matrix3D other);

  Matrix3D multiplyElementWise(Matrix3D other);

  void normalize(double mean, double std);

  Matrix3D transpose();

  Matrix3D multiply(double value);

  Matrix3D sumColumns();

  double get(int depth, int row, int col);

  void set(int depth, int row, int col, double value);

  int getDepth();

  int getRows();

  int getCols();

  Matrix3D sum(int axis);

  Matrix3D sumOverDepth();

  Matrix3D sumOverRows();

  Matrix3D sumOverCols();
}
