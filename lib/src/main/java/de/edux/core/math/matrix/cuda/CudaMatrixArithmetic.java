package de.edux.core.math.matrix.cuda;

import de.edux.core.math.IMatrixArithmetic;
import de.edux.core.math.IMatrixProduct;
import de.edux.core.math.IMatrixVectorProduct;
import de.edux.core.math.matrix.cuda.operations.CudaMatrixProduct;
import de.edux.core.math.matrix.cuda.operations.CudaMatrixVectorProduct;
import de.edux.core.math.matrix.parallel.operations.MatrixProduct;
import de.edux.core.math.matrix.parallel.operations.MatrixVectorProduct;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CudaMatrixArithmetic implements IMatrixArithmetic {
  private static CudaMatrixArithmetic instance;
  private static final Logger LOG = LoggerFactory.getLogger(CudaMatrixArithmetic.class);
  private final IMatrixProduct matrixProduct;
  private final IMatrixVectorProduct matrixVectorProduct;

  private CudaMatrixArithmetic() {
    if (!isCudaAvailable()) {
      this.matrixProduct = new MatrixProduct();
      this.matrixVectorProduct = new MatrixVectorProduct();
    } else {
      this.matrixProduct = new CudaMatrixProduct();
      this.matrixVectorProduct = new CudaMatrixVectorProduct();
    }
  }

  private boolean isCudaAvailable() {
    LOG.info("Checking for CUDA availability.");
    try {
      int[] count = new int[1];
      JCuda.cudaGetDeviceCount(count);
      printCudaDeviceInformation();
      return true;
    } catch (Throwable e) {
      LOG.warn("CUDA is not available. Falling back to CPU implementation.", e);
      return false;
    }
  }

  private void printCudaDeviceInformation() {
    int[] deviceCount = new int[1];
    JCuda.cudaGetDeviceCount(deviceCount);

    LOG.info("Available CUDA devices : {}", deviceCount[0]);

    for (int i = 0; i < deviceCount[0]; i++) {
      cudaDeviceProp deviceProperties = new cudaDeviceProp();
      JCuda.cudaGetDeviceProperties(deviceProperties, i);

      LOG.info("::::::::::::::: CUDA device properties for device {} :::::::::::::::", i);
      LOG.info("Device {}: {}", i, deviceProperties.getName());
      LOG.info("  Total global memory: {}", deviceProperties.totalGlobalMem);
      LOG.info("  Shared memory per block: {}", deviceProperties.sharedMemPerBlock);
      LOG.info("  Registers per block: {}", deviceProperties.regsPerBlock);
      LOG.info("  Warp size: {}", deviceProperties.warpSize);
      LOG.info("  Memory Pitch: {}", deviceProperties.memPitch);
      LOG.info("  Max threads per block: {}", deviceProperties.maxThreadsPerBlock);
      LOG.info("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
    }
  }

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {
    if (matrixA == null || matrixB == null) {
      throw new IllegalArgumentException("Matrices must not be null.");
    }
    if (matrixA.length == 0 || matrixB.length == 0) {
      throw new IllegalArgumentException("Matrices must not be empty.");
    }
    if (matrixA[0].length != matrixB.length) {
      throw new IllegalArgumentException("Matrix A columns must match Matrix B rows.");
    }

    return matrixProduct.multiply(matrixA, matrixB);
  }

  @Override
  public double[] multiply(double[][] matrix, double[] vector) {
    if (matrix.length == 0 || matrix[0].length == 0 || vector.length == 0) {
      throw new IllegalArgumentException("Matrix and vector must not be empty.");
    }
    if (matrix[0].length != vector.length) {
      throw new IllegalArgumentException("Matrix columns and vector size do not match.");
    }
    return matrixVectorProduct.multiply(matrix, vector);
  }
  public static CudaMatrixArithmetic getInstance() {
    if(instance == null) {
      instance =  new CudaMatrixArithmetic();
    }
    return instance;
  }
}
