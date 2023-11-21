__kernel void matrixProduct (__global double *A, __global double *B, __global double *C, const int aCols, const int bCols) {
    int j,k;
    int i = get_global_id(0);

    int localCopyA[SIZE];
    for (k = 0; k < SIZE; k++) {
        localCopyA[k] = A[i*SIZE+k];
    }

    double result;
    for (j = 0; j < bCols; j++) {
        result = 0;
        for (k = 0; k < aCols; k++) {
            result += localCopyA[k] * B[k*bCols+j];
        }
        C[i*bCols+j] = result;
    }
}