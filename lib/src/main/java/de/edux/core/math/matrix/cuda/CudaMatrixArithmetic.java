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

  private static final Logger LOG = LoggerFactory.getLogger(CudaMatrixArithmetic.class);
  private final IMatrixProduct matrixProduct;
  private final IMatrixVectorProduct matrixVectorProduct;

  public CudaMatrixArithmetic() {
    if (!isCudaAvailable()) {
      this.matrixProduct = new MatrixProduct();
      this.matrixVectorProduct = new MatrixVectorProduct();
    } else {
      this.matrixProduct = new CudaMatrixProduct();
      this.matrixVectorProduct = new CudaMatrixVectorProduct();
    }
  }

  private boolean isCudaAvailable() {
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
    return matrixProduct.multiply(matrixA, matrixB);
  }

  @Override
  public double[] multiply(double[][] matrix, double[] vector) {
    return matrixVectorProduct.multiply(matrix, vector);
  }
}
