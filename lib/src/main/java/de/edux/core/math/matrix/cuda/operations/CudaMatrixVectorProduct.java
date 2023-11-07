package de.edux.core.math.matrix.cuda.operations;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;

import de.edux.core.math.IMatrixVectorProduct;
import de.edux.core.math.matrix.cuda.CUDAKernelUser;
import java.io.File;
import java.util.List;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.driver.CUfunction;

public class CudaMatrixVectorProduct implements IMatrixVectorProduct, CUDAKernelUser {

  static {
    JCudaDriver.setExceptionsEnabled(true);
    cuInit(0);
    CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    cuCtxCreate(context, 0, device);
  }

  @Override
  public double[] multiply(double[][] matrix, double[] vector) {
    int numRows = matrix.length;
    int numCols = matrix[0].length;

    if (numCols != vector.length) {
      throw new IllegalArgumentException("Matrix columns and vector size do not match.");
    }

    CUfunction function = loadKernel();

    double[] hostMatrix = flatten(matrix);
    double[] hostVector = vector.clone();
    double[] hostOutput = new double[numRows];

    CUdeviceptr deviceMatrix = new CUdeviceptr();
    cuMemAlloc(deviceMatrix, hostMatrix.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceMatrix, Pointer.to(hostMatrix), hostMatrix.length * Sizeof.DOUBLE);

    CUdeviceptr deviceVector = new CUdeviceptr();
    cuMemAlloc(deviceVector, hostVector.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceVector, Pointer.to(hostVector), hostVector.length * Sizeof.DOUBLE);

    CUdeviceptr deviceOutput = new CUdeviceptr();
    cuMemAlloc(deviceOutput, hostOutput.length * Sizeof.DOUBLE);

    Pointer kernelParameters =
        Pointer.to(
            Pointer.to(deviceMatrix),
            Pointer.to(deviceVector),
            Pointer.to(deviceOutput),
            Pointer.to(new int[] {numRows}),
            Pointer.to(new int[] {numCols}));

    int blockSize = 256; // This should be tuned according to your hardware capability
    int gridSize = (int) Math.ceil((double) numRows / blockSize);
    cuLaunchKernel(
        function,
        gridSize,
        1,
        1, // Grid dimension
        blockSize,
        1,
        1, // Block dimension
        0,
        null, // Shared memory size and stream
        kernelParameters,
        null // Kernel- and extra parameters
        );
    cuCtxSynchronize();

    cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, hostOutput.length * Sizeof.DOUBLE);

    List<CUdeviceptr> list = List.of(deviceMatrix, deviceVector, deviceOutput);
    cleanUp(list);

    return hostOutput;
  }

  private void cleanUp(List<CUdeviceptr> devicePtrs) {
    for (CUdeviceptr devicePtr : devicePtrs) {
      cuMemFree(devicePtr);
    }
  }

  private double[] flatten(double[][] matrix) {
    int rows = matrix.length;
    int cols = matrix[0].length;
    double[] flat = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(matrix[i], 0, flat, i * cols, cols);
    }
    return flat;
  }

  @Override
  public CUfunction loadKernel() {
    String ptxFileName = "cuda_kernels" + File.separator + "matrixVectorMultiplicationKernel.ptx";
    CUmodule module = new CUmodule();
    cuModuleLoad(module, ptxFileName);
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "matrixVectorMultiplicationKernel");
    return function;
  }
}
