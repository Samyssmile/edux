extern "C"
__global__ void matrixVectorMultiplicationKernel(double *matrix, double *vector, double *result, int numRows, int numCols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numRows) {
    double sum = 0.0;
    for (int col = 0; col < numCols; ++col) {
      sum += matrix[row * numCols + col] * vector[col];
    }
    result[row] = sum;
  }
}
