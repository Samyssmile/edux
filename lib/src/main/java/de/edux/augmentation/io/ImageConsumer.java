package de.edux.augmentation.io;

import de.edux.augmentation.core.AugmentationSequence;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import javax.imageio.ImageIO;

public class ImageConsumer implements Runnable {
  private final BlockingQueue<ImageWithName> queue;
  private final AugmentationSequence augmentationSequence;
  private final String outputDirectoryPath;

  public ImageConsumer(
      BlockingQueue<ImageWithName> queue,
      AugmentationSequence augmentationSequence,
      String outputDirectoryPath) {
    this.queue = queue;
    this.augmentationSequence = augmentationSequence;
    this.outputDirectoryPath = outputDirectoryPath;
  }

  @Override
  public void run() {
    try {
      while (true) {
        ImageWithName image = queue.poll(2000, TimeUnit.MILLISECONDS);
        BufferedImage augmentedImage = augmentationSequence.applyTo(image.image());
        processImage(augmentedImage, image.fileName());
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      System.out.println("Consumer wurde unterbrochen.");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private void processImage(BufferedImage image, String fileName) throws IOException {
    File outputFile = Paths.get(outputDirectoryPath, fileName).toFile();

    File outputDir = outputFile.getParentFile();
    if (!outputDir.exists()) {
      outputDir.mkdirs();
    }

    ImageIO.write(image, "png", outputFile);
    System.out.println("Bild gespeichert: " + outputFile.getAbsolutePath());
  }
}
