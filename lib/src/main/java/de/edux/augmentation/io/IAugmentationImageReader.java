package de.edux.augmentation.io;

import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.Stream;

public interface IAugmentationImageReader {

  /**
   * Reads an image from the given path.
   *
   * @return The BufferedImage read from the file.
   * @throws IOException If an error occurs during reading the image.
   */
  Stream<Path> readBatchOfImages(String directoryPath, int width, int height) throws IOException;
}
