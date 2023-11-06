extern "C"
__global__ void matrixMultiply(double *A, double *B, double *C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        double sum = 0;
        for (int i = 0; i < numAColumns; ++i) {
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}
