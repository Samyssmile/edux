package de.edux.core.math.matrix.cuda;

import jcuda.driver.CUfunction;

public interface CUDAKernelUser {
  CUfunction loadKernel();
}
