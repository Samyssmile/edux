package de.edux.augmentation.io;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import javax.imageio.ImageIO;

public class AugmentationImageReader implements IAugmentationImageReader {
  public Stream<BufferedImage> readBatchOfImages(Path path, int batchSize) throws IOException {
    return Files.list(path)
        .parallel()
        .filter(Files::isRegularFile)
        .filter(p -> p.toString().endsWith(".jpg") || p.toString().endsWith(".png"))
        .map(
            p -> {
              try {
                return ImageIO.read(p.toFile());
              } catch (IOException e) {
                throw new RuntimeException(e);
              }
            });
  }
}
