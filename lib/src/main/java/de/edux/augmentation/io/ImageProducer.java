package de.edux.augmentation.io;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.BlockingQueue;
import javax.imageio.ImageIO;

/**
 * Handles the production of images from a specified directory, queuing them for further processing.
 *
 * <p>This class is designed to work as part of a producer-consumer pattern, where it acts as a
 * producer that reads images from a directory and puts them into a shared queue. It is intended to
 * be used in a multithreaded environment, typically with multiple consumers processing the queued
 * images.
 */
public class ImageProducer implements Runnable {

  private final BlockingQueue<ImageWithName> queue;
  private final String directoryPathString;
  private final int numberOfConsumers;
  private AugmentationImageReader augmentationImageReader = new AugmentationImageReader();

  /**
   * Constructs an ImageProducer with the specified queue, directory path, and number of consumers.
   *
   * <p>This constructor initializes the producer with the necessary details to start reading and
   * queuing images from the given directory.
   *
   * @param queue The queue into which images with names are to be put.
   * @param directoryPathString The path of the directory from which to read images.
   * @param numberOfConsumers The number of consumers that will be processing the images.
   * @throws IOException If an I/O error occurs while reading from the directory.
   */
  public ImageProducer(
      BlockingQueue<ImageWithName> queue, String directoryPathString, int numberOfConsumers)
      throws IOException {
    this.queue = queue;
    this.directoryPathString = directoryPathString;
    this.numberOfConsumers = numberOfConsumers;
  }

  /**
   * The run method invoked when the thread starts.
   *
   * <p>This method reads image paths from the specified directory and places the images along with
   * their names into the queue. It handles access denial and general I/O exceptions, printing an
   * error message in these cases. The method also ensures that the images are read and queued in a
   * way that allows multiple consumers to process them concurrently.
   */
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
