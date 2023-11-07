package de.edux.core.math.matrix.cuda.operations;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;

import de.edux.core.math.IMatrixProduct;
import java.io.File;
import java.util.List;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

public class CudaMatrixProduct implements IMatrixProduct {

  static {
    JCudaDriver.setExceptionsEnabled(true);
    cuInit(0);
    CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    cuCtxCreate(context, 0, device);
  }

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {
    int aRows = matrixA.length;
    int aCols = matrixA[0].length;
    int bCols = matrixB[0].length;

    if (aCols != matrixB.length) {
      throw new IllegalArgumentException("Inner dimensions do not match.");
    }

    String ptxFileName = preparePtxFile();

    CUmodule module = new CUmodule();
    cuModuleLoad(module, ptxFileName);
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "matrixMultiply");

    double[] hostInputA = flatten(matrixA);
    double[] hostInputB = flatten(matrixB);
    double[] hostOutput = new double[aRows * bCols];

    CUdeviceptr deviceInputA = new CUdeviceptr();
    cuMemAlloc(deviceInputA, hostInputA.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA), hostInputA.length * Sizeof.DOUBLE);

    CUdeviceptr deviceInputB = new CUdeviceptr();
    cuMemAlloc(deviceInputB, hostInputB.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB), hostInputB.length * Sizeof.DOUBLE);

    CUdeviceptr deviceOutput = new CUdeviceptr();
    cuMemAlloc(deviceOutput, hostOutput.length * Sizeof.DOUBLE);

    Pointer kernelParameters =
        Pointer.to(
            Pointer.to(deviceInputA),
            Pointer.to(deviceInputB),
            Pointer.to(deviceOutput),
            Pointer.to(new int[] {aRows}),
            Pointer.to(new int[] {aCols}),
            Pointer.to(new int[] {bCols}));

    int blockSizeX = 16;
    int blockSizeY = 16;
    int gridSizeX = (int) Math.ceil((double) aRows / blockSizeX);
    int gridSizeY = (int) Math.ceil((double) bCols / blockSizeY);
    cuLaunchKernel(
        function,
        gridSizeX,
        gridSizeY,
        1, // Grid dimension
        blockSizeX,
        blockSizeY,
        1, // Block dimension
        0,
        null, // Shared memory size and stream
        kernelParameters,
        null // Kernel- and extra parameters
        );
    cuCtxSynchronize();

    cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, hostOutput.length * Sizeof.DOUBLE);

    List<CUdeviceptr> list = List.of(deviceInputA, deviceInputB, deviceOutput);
    cleanUp(list);

    double[][] result = new double[aRows][bCols];
    int index = 0;
    for (int i = 0; i < aRows; i++) {
      for (int j = 0; j < bCols; j++) {
        result[i][j] = hostOutput[index++];
      }
    }

    return result;
  }

  private void cleanUp(List<CUdeviceptr> cUdeviceptrs) {
    for (CUdeviceptr cUdeviceptr : cUdeviceptrs) {
      cuMemFree(cUdeviceptr);
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

  private String preparePtxFile() {
    return "cuda_kernels" + File.separator + "matrixMultiplicationKernel.ptx";
  }
}
