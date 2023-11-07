package de.edux.core.math.matrix.cuda.exceptions;

public class CudaNotAvailableException extends Exception {
  public CudaNotAvailableException(String message) {
    super("CUDA is not available: " + message);
  }
}
