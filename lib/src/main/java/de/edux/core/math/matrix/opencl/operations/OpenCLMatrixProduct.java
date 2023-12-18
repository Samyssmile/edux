package de.edux.core.math.matrix.opencl.operations;

import static org.jocl.CL.*;

import de.edux.core.math.IMatrixProduct;
import de.edux.opencl.OpenCLSetup;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import org.jocl.*;

public class OpenCLMatrixProduct implements IMatrixProduct, OpenCLKernelUser {
  private final cl_context context;
  private final cl_command_queue commandQueue;

  public OpenCLMatrixProduct(OpenCLSetup openCLSetup) {
    cl_platform_id platform = openCLSetup.getPlatform();
    cl_device_id device = openCLSetup.getDevice();

    cl_context_properties contextProperties = new cl_context_properties();
    contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
    context = clCreateContext(contextProperties, 1, new cl_device_id[] {device}, null, null, null);

    cl_queue_properties properties = new cl_queue_properties();
    commandQueue = clCreateCommandQueueWithProperties(context, device, properties, null);
  }

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {
    int aRows = matrixA.length;
    int aCols = matrixA[0].length;
    int bCols = matrixB[0].length;

    double[] hostInputA = flatten(matrixA);
    double[] hostInputB = flatten(matrixB);
    double[] hostOutput = new double[aRows * bCols];

    Pointer pointerToInputA = Pointer.to(hostInputA);
    Pointer pointerToInputB = Pointer.to(hostInputB);
    Pointer pointerToOutput = Pointer.to(hostOutput);

    cl_mem srcMemA =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) Sizeof.cl_double * hostInputA.length,
            pointerToInputA,
            null);
    cl_mem srcMemB =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) Sizeof.cl_double * hostInputB.length,
            pointerToInputB,
            null);

    cl_mem dstMem =
        clCreateBuffer(
            context, CL_MEM_READ_WRITE, (long) Sizeof.cl_double * hostOutput.length, null, null);

    String kernelSource = "#define SIZE " + aCols + "\n" + loadKernelSource();

    cl_program program =
        clCreateProgramWithSource(context, 1, new String[] {kernelSource}, null, null);

    clBuildProgram(program, 0, null, null, null, null);
    cl_kernel kernel = clCreateKernel(program, "matrixProduct", null);

    clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(srcMemA));
    clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(srcMemB));
    clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(dstMem));
    clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {aCols}));
    clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {bCols}));

    long[] global_work_size = new long[] {aCols};

    clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, null, 0, null, null);

    clEnqueueReadBuffer(
        commandQueue,
        dstMem,
        CL_TRUE,
        0,
        (long) hostOutput.length * Sizeof.cl_double,
        pointerToOutput,
        0,
        null,
        null);

    clReleaseMemObject(srcMemA);
    clReleaseMemObject(srcMemB);
    clReleaseMemObject(dstMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    double[][] result = new double[aRows][bCols];
    int index = 0;
    for (int i = 0; i < aRows; i++) {
      for (int j = 0; j < bCols; j++) {
        result[i][j] = hostOutput[index++];
      }
    }

    return result;
  }

  private double[] flatten(double[][] matrix) {
    int rows = matrix.length;
    int cols = matrix[0].length;
    double[] flat = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(matrix[i], 0, flat, i * cols, cols);
    }
    return flat;
  }

  public String loadKernelSource() {
    BufferedReader bufferedReader = null;
    String fileName = "../kernels/opencl/matrixMultiplicationKernel.cl";

    try {
      bufferedReader = new BufferedReader(new FileReader(fileName));
      StringBuilder stringBuilder = new StringBuilder();
      String line;
      while (true) {
        line = bufferedReader.readLine();
        if (line == null) {
          break;
        }
        stringBuilder.append(line).append("\n");
      }
      return stringBuilder.toString();
    } catch (IOException e) {
      System.out.println(e.getMessage());
      return "";
    } finally {
      if (bufferedReader != null) {
        try {
          bufferedReader.close();
        } catch (IOException e) {
          System.out.println(e.getMessage());
        }
      }
    }
  }
}
