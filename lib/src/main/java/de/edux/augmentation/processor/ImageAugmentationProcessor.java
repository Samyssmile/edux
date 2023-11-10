package de.edux.augmentation.processor;

import de.edux.augmentation.core.AbstractAugmentation;
import de.edux.augmentation.core.AugmentationSequence;
import de.edux.augmentation.io.IAugmentationImageReader;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.*;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.imageio.ImageIO;

public class ImageAugmentationProcessor {

  private final AugmentationSequence augmentationSequence;
  private final IAugmentationImageReader imageReader;
  private final Path inputDirectory;
  private final Path outputDirectory;

  public ImageAugmentationProcessor(
      AugmentationSequence augmentationSequence,
      IAugmentationImageReader imageReader,
      Path inputDirectory,
      Path outputDirectory) {
    this.augmentationSequence = augmentationSequence;
    this.imageReader = imageReader;
    this.inputDirectory = inputDirectory;
    this.outputDirectory = outputDirectory;
  }

  public void processImages() throws IOException {
    try (Stream<Path> paths = imageReader.readImagePathsAsStream(inputDirectory.toString())) {
      // Konvertiere den Stream in eine Liste von Pfaden, um sicherzustellen, dass die Dateiliste
      // passt
      List<Path> pathList = paths.filter(Files::isRegularFile).collect(Collectors.toList());

      // Definieren Sie die Anzahl der Threads basierend auf den verfügbaren Ressourcen
      int numberOfThreads = 3;

      // Erstellen Sie einen ForkJoinPool mit der angegebenen Anzahl von Threads
      ForkJoinPool customThreadPool = new ForkJoinPool(3);
      try {
        // Verwenden Sie den benutzerdefinierten ThreadPool, um die Augmentationen parallel
        // auszuführen
        customThreadPool
            .submit(() -> pathList.parallelStream().forEach(this::augmentAndSaveImage))
            .get(); // Warten Sie, bis alle Aufgaben abgeschlossen sind
      } catch (InterruptedException | ExecutionException e) {
        e.printStackTrace();
      } finally {
        customThreadPool.shutdown(); // Schließen Sie den ThreadPool nach der Verarbeitung
      }
    }
  }

  private void augmentAndSaveImage(Path imagePath) {
    try {
      BufferedImage image = AbstractAugmentation.readImage(imagePath);
      BufferedImage augmentedImage = augmentationSequence.applyTo(image);
      Path outputPath = outputDirectory.resolve(inputDirectory.relativize(imagePath));
      Files.createDirectories(
          outputPath.getParent()); // Ensure the output directory structure is created
      ImageIO.write(
          augmentedImage,
          "png",
          outputPath.toFile()); // Replace "jpg" with the appropriate format if needed
    } catch (IOException e) {
      e.printStackTrace(); // Handle the exception according to your needs
    }
  }
}
