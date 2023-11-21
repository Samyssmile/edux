package de.edux.opencl.exceptions;

public class OpenCLNotAvailableException extends Exception {
  public OpenCLNotAvailableException(String message) {
    super("OpenCL is not available: " + message);
  }
}
