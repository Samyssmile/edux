package de.edux.opencl;

import de.edux.opencl.exceptions.OpenCLNotAvailableException;
import de.edux.opencl.util.OpenCLDeviceQuery;
import java.util.Map;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

public class OpenCLSetup {
  private final Map<cl_platform_id, cl_device_id> setup;

  public OpenCLSetup(int deviceNumber) throws OpenCLNotAvailableException {
    if (OpenCLDeviceQuery.isOpenCLAvailable()) {
      setup = OpenCLDeviceQuery.getOpenCLDeviceMapping().get(deviceNumber);
    } else {
      throw new OpenCLNotAvailableException("Set up a fallback method.");
    }
  }

  public cl_device_id getDevice() {
    return setup.values().stream().toList().get(0);
  }

  public cl_platform_id getPlatform() {
    return setup.keySet().stream().toList().get(0);
  }
}
