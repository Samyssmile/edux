package de.edux;

import jcuda.*;
import jcuda.runtime.*;
import jcuda.driver.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.stream.Stream;

public class CudaImageResizer {

    public static void main(String[] args) {
        String inputDirectoryPath = "C:\\Users\\windo\\Pictures\\dataset\\class\\tricorn";
        String outputDirectoryPath = "resize_cuda";
        int newWidth = 100;
        int newHeight = 100;

        try (Stream<Path> paths = Files.walk(Paths.get(inputDirectoryPath))) {
            paths.filter(Files::isRegularFile)
                    .forEach(path -> {
                        try {
                            // Laden Sie das Bild in ein Byte-Array
                            byte[] inputImage = loadImage(path.toFile());

                            // Erstellen Sie ein Byte-Array, um das Ausgabebild aufzunehmen
                            byte[] outputImage = new byte[newWidth * newHeight * 3]; // Angenommen, es handelt sich um ein RGB-Bild

                            // Rufen Sie die Methode auf, um das Bild zu verkleinern
                            resizeImage(inputImage,  4000, 4000, outputImage, newWidth, newHeight);

                            // Speichern Sie das Ausgabebild
                            Path outputPath = Paths.get(outputDirectoryPath, path.getFileName().toString());
                            saveImage(outputImage, outputPath.toFile(), newWidth, newHeight);

                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static byte[] loadImage(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "png", baos); // Sie können "png" durch Ihr spezifisches Bildformat ersetzen, falls nötig
        return baos.toByteArray();
    }

    private static void saveImage(byte[] imageData, File file, int width, int height) throws IOException {
        // Hier nehmen wir an, dass das Byte-Array ein RGB-Array ist
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        image.getRaster().setDataElements(0, 0, width, height, imageData);

        // Stellen Sie sicher, dass der Ausgabeordner existiert
        Path outputPath = file.toPath().getParent();
        if (outputPath != null && Files.notExists(outputPath)) {
            Files.createDirectories(outputPath);
        }

        ImageIO.write(image, "png", file); // Sie können "png" durch Ihr spezifisches Bildformat ersetzen, falls nötig
    }
    public static void resizeImage(byte[] inputImage, int originalWidth, int originalHeight, byte[] outputImage, int newWidth, int newHeight) {
        // Initialisiere JCuda
        JCuda.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        JCudaDriver.cuDeviceGet(dev, 0);
        JCudaDriver.cuCtxCreate(pctx, 0, dev);

        // Lade den Kernel
        CUmodule module = new CUmodule();
    JCudaDriver.cuModuleLoad(
        module,
        "C:\\Users\\windo\\Documents\\projekte\\edux\\lib\\src\\main\\java\\de\\edux\\scaleImageKernel.ptx");
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "scaleImageKernel");

        // Alloziere Speicher auf der GPU
        CUdeviceptr deviceInput = new CUdeviceptr();
        CUdeviceptr deviceOutput = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(deviceInput, inputImage.length);
        JCudaDriver.cuMemAlloc(deviceOutput, outputImage.length);

        // Kopiere die Daten zum Device
        JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(inputImage), inputImage.length);

        // Setze die Kernel-Parameter
        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceInput),
                Pointer.to(deviceOutput),
                Pointer.to(new int[]{originalWidth}),
                Pointer.to(new int[]{originalHeight}),
                Pointer.to(new int[]{newWidth}),
                Pointer.to(new int[]{newHeight})
        );

        // Rufe den Kernel auf
        int blockSizeX = 16;
        int blockSizeY = 16;
        int gridSizeX = (newWidth + blockSizeX - 1) / blockSizeX;
        int gridSizeY = (newHeight + blockSizeY - 1) / blockSizeY;
        JCudaDriver.cuLaunchKernel(function,
                gridSizeX, gridSizeY, 1,      // Grid dimension
                blockSizeX, blockSizeY, 1,    // Block dimension
                0, null,                      // Shared memory size and stream
                kernelParameters, null        // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();

        // Kopiere die veränderten Daten zurück zum Host
        JCudaDriver.cuMemcpyDtoH(Pointer.to(outputImage), deviceOutput, outputImage.length);

        // Ressourcen freigeben
        JCudaDriver.cuMemFree(deviceInput);
        JCudaDriver.cuMemFree(deviceOutput);
        JCudaDriver.cuModuleUnload(module);
        JCudaDriver.cuCtxDestroy(pctx);
    }
}
