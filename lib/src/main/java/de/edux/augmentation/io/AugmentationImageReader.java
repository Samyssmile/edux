package de.edux.augmentation.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

/**
 * Implementation of {@link IAugmentationImageReader} to read image paths from a specified
 * directory.
 *
 * <p>This class is used to read and stream the paths of images from a given directory. It's part of
 * the image augmentation process where image file paths are required for further processing.
 */
public class AugmentationImageReader implements IAugmentationImageReader {

  /**
   * Reads all image file paths from a specified directory and returns them as a stream.
   *
   * <p>This method walks through the directory denoted by the given path and collects the paths of
   * all files found. It is designed to efficiently handle large directories by streaming the paths
   * instead of storing them in memory.
   *
   * @param directoryPath The path to the directory from which image paths are to be read.
   * @return A {@link Stream} of {@link Path} objects, each representing a path to a file in the
   *     directory.
   * @throws IOException If an I/O error is encountered when accessing the directory.
   */
  public Stream<Path> readImagePathsAsStream(String directoryPath) throws IOException {
    return Files.walk(Paths.get(directoryPath));
  }
}
