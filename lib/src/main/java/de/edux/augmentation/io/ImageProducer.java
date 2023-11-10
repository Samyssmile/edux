package de.edux.augmentation.io;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.BlockingQueue;
import javax.imageio.ImageIO;

public class ImageProducer implements Runnable {

  private final BlockingQueue<ImageWithName> queue;
  private final String directoryPathString;
  private final int numberOfConsumers;
  private AugmentationImageReader augmentationImageReader = new AugmentationImageReader();

  public ImageProducer(
      BlockingQueue<ImageWithName> queue, String directoryPathString, int numberOfConsumers)
      throws IOException {
    this.queue = queue;
    this.directoryPathString = directoryPathString;
    this.numberOfConsumers = numberOfConsumers;
  }

  @Override
  public void run() {
    try {
      this.augmentationImageReader
          .readImagePathsAsStream(directoryPathString)
          .forEach(
              path -> {
                try (InputStream is = Files.newInputStream(Paths.get(path.toString()))) {
                  queue.put(new ImageWithName(ImageIO.read(is), path.getFileName().toString()));
                } catch (AccessDeniedException ex) {
                  System.out.println("Access denied: " + path.toString());
                } catch (IOException | InterruptedException e) {
                  e.printStackTrace();
                }
              });
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
