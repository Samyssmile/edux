package de.edux.augmentation.io;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.Stream;

public interface IAugmentationImageReader {

  /**
   * Reads an image from the given path.
   *
   * @param path The path of the image file to be read.
   * @return The BufferedImage read from the file.
   * @throws IOException If an error occurs during reading the image.
   */
  Stream<BufferedImage> readBatchOfImages(Path path, int batchSize) throws IOException;
}
