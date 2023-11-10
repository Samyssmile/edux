package de.edux.augmentation.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class AugmentationImageReader implements IAugmentationImageReader {
  public Stream<Path> readImagePathsAsStream(String directoryPath) throws IOException {
    return Files.walk(Paths.get(directoryPath));
  }
}
