package de.edux.opencl.util;

import static org.jocl.CL.*;

import java.nio.*;
import org.jocl.*;

class DeviceProperties {

  static String getDeviceVendor(cl_device_id device) {
    return getString(device, CL_DEVICE_VENDOR);
  }

  static String getDeviceName(cl_device_id device) {
    return getString(device, CL_DEVICE_NAME);
  }

  static int getMaxComputeUnits(cl_device_id device) {
    return getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
  }

  static long getMaxWorkItemDimensions(cl_device_id device) {
    return getLong(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
  }

  static long[] getMaxWorkItemSizes(cl_device_id device) {
    return getSizes(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
  }

  static long getMaxWorkGroupSize(cl_device_id device) {
    return getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
  }

  static long getMaxClockFrequency(cl_device_id device) {
    return getLong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
  }

  static long getMaxGlobalMemorySize(cl_device_id device) {
    return getLong(device, CL_DEVICE_GLOBAL_MEM_SIZE);
  }

  static long getMaxLocalMemorySize(cl_device_id device) {
    return getLong(device, CL_DEVICE_LOCAL_MEM_SIZE);
  }

  static long getMaxMemoryAllocationSize(cl_device_id device) {
    return getLong(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
  }

  private static int getInt(cl_device_id device, int paramName) {
    return getInts(device, paramName, 1)[0];
  }

  private static int[] getInts(cl_device_id device, int paramName, int numValues) {
    int[] values = new int[numValues];
    clGetDeviceInfo(device, paramName, (long) Sizeof.cl_int * numValues, Pointer.to(values), null);
    return values;
  }

  private static long getLong(cl_device_id device, int paramName) {
    return getLongs(device, paramName, 1)[0];
  }

  private static long[] getLongs(cl_device_id device, int paramName, int numValues) {
    long[] values = new long[numValues];
    clGetDeviceInfo(device, paramName, (long) Sizeof.cl_long * numValues, Pointer.to(values), null);
    return values;
  }

  private static String getString(cl_device_id device, int paramName) {
    long[] size = new long[1];
    clGetDeviceInfo(device, paramName, 0, null, size);

    byte[] buffer = new byte[(int) size[0]];
    clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

    return new String(buffer, 0, buffer.length - 1);
  }

  private static long getSize(cl_device_id device, int paramName) {
    return getSizes(device, paramName, 1)[0];
  }

  private static long[] getSizes(cl_device_id device, int paramName, int numValues) {
    ByteBuffer buffer =
        ByteBuffer.allocate(numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
    clGetDeviceInfo(device, paramName, (long) Sizeof.size_t * numValues, Pointer.to(buffer), null);
    long[] values = new long[numValues];
    if (Sizeof.size_t == 4) {
      for (int i = 0; i < numValues; i++) {
        values[i] = buffer.getInt(i * Sizeof.size_t);
      }
    } else {
      for (int i = 0; i < numValues; i++) {
        values[i] = buffer.getLong(i * Sizeof.size_t);
      }
    }
    return values;
  }
}
