package de.edux.augmentation.io;

import de.edux.augmentation.core.AugmentationSequence;
import java.io.IOException;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ImageProcessingManager {
  private final int numberOfConsumers;
  private final String inputDirectory;
  private final AugmentationSequence augmentationSequence;
  private final String outputDirectory;

  public ImageProcessingManager(
      String inputDirectoryPath,
      int numberOfConsumers,
      AugmentationSequence augmentationSequence,
      String outputDirectory) {
    this.inputDirectory = inputDirectoryPath;
    this.numberOfConsumers = numberOfConsumers;
    this.augmentationSequence = augmentationSequence;
    this.outputDirectory = outputDirectory;
  }

  public void processImages() throws InterruptedException, IOException {
    BlockingQueue<ImageWithName> queue = new ArrayBlockingQueue<>(1);

    Thread producerThread = new Thread(new ImageProducer(queue, inputDirectory, numberOfConsumers));
    producerThread.start();

    ExecutorService executorService = Executors.newFixedThreadPool(numberOfConsumers);
    for (int i = 0; i < numberOfConsumers; i++) {
      executorService.submit(new ImageConsumer(queue, augmentationSequence, outputDirectory));
    }

    producerThread.join();
    executorService.shutdown();
    executorService.awaitTermination(2000, TimeUnit.MILLISECONDS);
  }
}
