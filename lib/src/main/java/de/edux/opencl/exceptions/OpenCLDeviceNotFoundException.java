package de.edux.opencl.exceptions;

public class OpenCLDeviceNotFoundException extends RuntimeException {
  public OpenCLDeviceNotFoundException() {
    super("No OpenCL devices were found.");
  }
}
