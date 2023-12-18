__kernel void matrixVectorProduct (__global double *matrix, __global double *vector, __global double *C, const int matrixCols, const int vectorLength) {
    int j,k;
    int i = get_global_id(0);

    int localCopyA[SIZE];
    for (k = 0; k < SIZE; k++) {
        localCopyA[k] = matrix[i*SIZE+k];
    }

    double result = 0;
    for (k = 0; k < matrixCols; k++) {
        result += localCopyA[k] * vector[k];
    }
    C[i] = result;
}