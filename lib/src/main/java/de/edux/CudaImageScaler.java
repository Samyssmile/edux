package de.edux;

import jcuda.*;
import jcuda.driver.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.*;

import static jcuda.driver.JCudaDriver.*;

public class CudaImageScaler {

  private static final String KERNEL_PATH =
      "C:\\Users\\windo\\Documents\\projekte\\edux\\lib\\src\\main\\java\\de\\edux\\scaleImageKernel.ptx"; // Angenommen, der Kernel ist als PTX vorbereitet

    public static void main(String[] args) {
        String inputDirectoryPath = "C:\\Users\\windo\\Pictures\\dataset\\class\\tricorn";
        String outputDirectoryPath = "resize_cuda";
        int newWidth = 100;
        int newHeight = 100;

        // Initialisiere JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        cuCtxCreate(pctx, 0, dev);

        // Kompiliere den CUDA-Kernel
        CUmodule module = new CUmodule();
        cuModuleLoad(module, KERNEL_PATH);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "scaleImageKernel");

        // Lade das Bild und bereite den Speicher vor
        Path inputPath = Paths.get(inputDirectoryPath);
        Path outputPath = Paths.get(outputDirectoryPath);

        try {
            Files.createDirectories(outputPath); // Erstelle den Ausgabeordner, falls nicht vorhanden

            Files.list(inputPath).forEach(file -> {
                try {
                    BufferedImage image = ImageIO.read(file.toFile());
                    int inputWidth = image.getWidth();
                    int inputHeight = image.getHeight();

                    // Konvertiere BufferedImage in ein lineares Array von Pixeln
                    int[] inputPixels = image.getRGB(0, 0, inputWidth, inputHeight, null, 0, inputWidth);

                    // Erstelle Ausgabe-Pixelarray
                    int[] outputPixels = new int[newWidth * newHeight];

                    // CUDA-Speicherzuweisungen
                    CUdeviceptr deviceInput = new CUdeviceptr();
                    cuMemAlloc(deviceInput, inputWidth * inputHeight * Sizeof.INT);
                    CUdeviceptr deviceOutput = new CUdeviceptr();
                    cuMemAlloc(deviceOutput, newWidth * newHeight * Sizeof.INT);

                    // Kopiere die Pixel-Daten ins Device
                    cuMemcpyHtoD(deviceInput, Pointer.to(inputPixels), inputWidth * inputHeight * Sizeof.INT);

                    // Berechne Grid- und Block-Größen
                    int blockSizeX = 16;
                    int blockSizeY = 16;
                    int gridSizeX = (newWidth + blockSizeX - 1) / blockSizeX;
                    int gridSizeY = (newHeight + blockSizeY - 1) / blockSizeY;

                    // Starte den Kernel
                    Pointer kernelParameters = Pointer.to(
                            Pointer.to(deviceInput),
                            Pointer.to(deviceOutput),
                            Pointer.to(new int[]{inputWidth}),
                            Pointer.to(new int[]{inputHeight}),
                            Pointer.to(new int[]{newWidth}),
                            Pointer.to(new int[]{newHeight})
                    );

                    cuLaunchKernel(function,
                            gridSizeX, gridSizeY, 1,      // Grid dimension
                            blockSizeX, blockSizeY, 1,    // Block dimension
                            0, null,                      // Shared memory size and stream
                            kernelParameters, null        // Kernel- and extra parameters
                    );
                    cuCtxSynchronize();

                    // Hole die skalierten Pixel-Daten zurück
                    cuMemcpyDtoH(Pointer.to(outputPixels), deviceOutput, newWidth * newHeight * Sizeof.INT);

                    // Erzeuge ein neues BufferedImage für das Ergebnis
                    BufferedImage outputImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
                    outputImage.setRGB(0, 0, newWidth, newHeight, outputPixels, 0, newWidth);

                    // Speichere das skalierte Bild
                    File outputFile = outputPath.resolve(file.getFileName().toString()).toFile();
                    ImageIO.write(outputImage, "png", outputFile);

                    // Speicher freigeben
                    cuMemFree(deviceInput);
                    cuMemFree(deviceOutput);

                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Aufräumen
            cuCtxDestroy(pctx);
        }
    }
}
