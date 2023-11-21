package de.edux.opencl.util;

public class PrintOpenCLDevices {
  public static void main(String[] args) {
    OpenCLDeviceQuery.printAvailableDevices();
    OpenCLDeviceQuery.printAvailableDevicesWithProperties();
  }
}
