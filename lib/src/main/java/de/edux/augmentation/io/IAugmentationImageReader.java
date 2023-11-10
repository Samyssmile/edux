package de.edux.augmentation.io;

import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.Stream;

public interface IAugmentationImageReader {

  Stream<Path> readImagePathsAsStream(String directoryPath) throws IOException;
}
