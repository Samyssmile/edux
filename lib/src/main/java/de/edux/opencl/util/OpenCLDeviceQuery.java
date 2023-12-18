package de.edux.opencl.util;

import static org.jocl.CL.*;
import static org.jocl.CL.clGetPlatformIDs;

import de.edux.opencl.exceptions.OpenCLDeviceNotFoundException;
import java.util.HashMap;
import java.util.Map;
import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenCLDeviceQuery {
  private static final Map<Integer, Map<cl_platform_id, cl_device_id>> openCLDeviceMapping =
      new HashMap<>();
  private static final Logger LOG = LoggerFactory.getLogger(OpenCLDeviceQuery.class);

  static {
    fillDevicesMap();
  }

  public static Map<Integer, Map<cl_platform_id, cl_device_id>> getOpenCLDeviceMapping() {
    return openCLDeviceMapping;
  }

  public static boolean isOpenCLAvailable() {
    return getDevicesCount() > 0;
  }

  public static void printAvailableDevices() {
    printDevices(false);
  }

  public static void printAvailableDevicesWithProperties() {
    printDevices(true);
  }

  private static void printDevices(boolean withProperties) {
    int devicesCount = getDevicesCount();
    for (int i = 1; i <= devicesCount; i++) {
      cl_device_id device = openCLDeviceMapping.get(i).values().stream().toList().get(0);
      if (withProperties) {
        printDeviceProperties(device, i);
      } else {
        printDevice(device, i);
      }
    }
  }

  private static void printDevice(cl_device_id device, int deviceNumber) {
    if (isOpenCLAvailable()) {
      LOG.info("Printing available devices:");
      LOG.info("  Device number {}: {}", deviceNumber, DeviceProperties.getDeviceName(device));
      System.out.println();
    } else {
      throw new OpenCLDeviceNotFoundException();
    }
  }

  private static void printDeviceProperties(cl_device_id device, int deviceNumber) {
    LOG.info(
        "::::::::::::::: OpenCL device properties for device number {} :::::::::::::::",
        deviceNumber);
    LOG.info(
        "Device {}: {}",
        DeviceProperties.getDeviceVendor(device),
        DeviceProperties.getDeviceName(device));
    LOG.info("Device max compute units: {}", DeviceProperties.getMaxComputeUnits(device));
    LOG.info(
        "Device max work item dimensions: {}", DeviceProperties.getMaxWorkItemDimensions(device));
    LOG.info(
        "Device max work item sizes: {} / {} / {}",
        DeviceProperties.getMaxWorkItemSizes(device)[0],
        DeviceProperties.getMaxWorkItemSizes(device)[1],
        DeviceProperties.getMaxWorkItemSizes(device)[2]);
    LOG.info("Device work group size: {}", DeviceProperties.getMaxWorkGroupSize(device));
    LOG.info("Device max clock frequency: {} MHz", DeviceProperties.getMaxClockFrequency(device));
    LOG.info(
        "Device max global memory size: {} MByte",
        (int) (DeviceProperties.getMaxGlobalMemorySize(device) / (1024 * 1024)));
    LOG.info(
        "Device max local memory size: {} KByte",
        (int) (DeviceProperties.getMaxLocalMemorySize(device) / 1024));
    LOG.info(
        "Device max memory allocation size: {} MByte",
        (int) (DeviceProperties.getMaxMemoryAllocationSize(device) / (1024 * 1024)));
    LOG.info("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
  }

  private static void fillDevicesMap() {
    cl_platform_id[] platforms = getOpenCLPlatformIDs();
    int deviceIndex = 1;
    for (int i = 0; i < platforms.length; i++) {
      cl_device_id[] devices = getOpenCLDeviceIDs(platforms[i]);
      for (int j = 0; j < devices.length; j++) {
        Map<cl_platform_id, cl_device_id> openCLSetupMap = new HashMap<>();
        openCLSetupMap.put(platforms[i], devices[j]);
        OpenCLDeviceQuery.openCLDeviceMapping.put(deviceIndex, openCLSetupMap);
        deviceIndex++;
      }
    }
  }

  private static cl_platform_id[] getOpenCLPlatformIDs() {
    int[] platformsCount = new int[1];
    clGetPlatformIDs(0, null, platformsCount);
    cl_platform_id[] platforms = new cl_platform_id[platformsCount[0]];
    clGetPlatformIDs(platformsCount[0], platforms, null);
    return platforms;
  }

  private static cl_device_id[] getOpenCLDeviceIDs(cl_platform_id platform) {
    int[] devicesCount = new int[1];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, devicesCount);
    cl_device_id[] devices = new cl_device_id[devicesCount[0]];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount[0], devices, null);
    return devices;
  }

  private static int getDevicesCount() {
    return openCLDeviceMapping.size();
  }
}
