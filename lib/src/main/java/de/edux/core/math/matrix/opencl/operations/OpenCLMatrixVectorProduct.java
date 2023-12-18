package de.edux.core.math.matrix.opencl.operations;

import static org.jocl.CL.*;

import de.edux.core.math.IMatrixVectorProduct;
import de.edux.opencl.OpenCLSetup;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import org.jocl.*;

public class OpenCLMatrixVectorProduct implements IMatrixVectorProduct, OpenCLKernelUser {

  private final cl_context context;
  private final cl_command_queue commandQueue;

  public OpenCLMatrixVectorProduct(OpenCLSetup openCLSetup) {
    cl_platform_id platform = openCLSetup.getPlatform();
    cl_device_id device = openCLSetup.getDevice();

    cl_context_properties contextProperties = new cl_context_properties();
    contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
    context = clCreateContext(contextProperties, 1, new cl_device_id[] {device}, null, null, null);

    cl_queue_properties properties = new cl_queue_properties();
    commandQueue = clCreateCommandQueueWithProperties(context, device, properties, null);
  }

  @Override
  public double[] multiply(double[][] matrix, double[] vector) {
    int matrixRows = matrix.length;
    int matrixCols = matrix[0].length;
    int vectorLength = vector.length;

    double[] hostInputMatrix = flatten(matrix);
    double[] hostOutput = new double[matrixRows];

    Pointer pointerToInputMatrix = Pointer.to(hostInputMatrix);
    Pointer pointerToInputVector = Pointer.to(vector);
    Pointer pointerToOutput = Pointer.to(hostOutput);

    cl_mem srcMemMatrix =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) Sizeof.cl_double * hostInputMatrix.length,
            pointerToInputMatrix,
            null);

    cl_mem srcMemVector =
        clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (long) Sizeof.cl_double * vectorLength,
            pointerToInputVector,
            null);

    cl_mem dstMem =
        clCreateBuffer(
            context, CL_MEM_READ_WRITE, (long) Sizeof.cl_double * hostOutput.length, null, null);

    String kernelSource = "#define SIZE " + matrixCols + "\n" + loadKernelSource();

    cl_program program =
        clCreateProgramWithSource(context, 1, new String[] {kernelSource}, null, null);

    clBuildProgram(program, 0, null, null, null, null);
    cl_kernel kernel = clCreateKernel(program, "matrixVectorProduct", null);

    clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(srcMemMatrix));
    clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(srcMemVector));
    clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(dstMem));
    clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {matrixCols}));
    clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] {vectorLength}));

    long[] global_work_size = new long[] {matrixCols};

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

    clReleaseMemObject(srcMemMatrix);
    clReleaseMemObject(srcMemVector);
    clReleaseMemObject(dstMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return hostOutput;
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
    String fileName = "../kernels/opencl/matrixVectorMultiplicationKernel.cl";

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
