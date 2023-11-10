package de.edux.augmentation.io;

import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.openjdk.jmh.annotations.Setup;

public class AugmentationIOTest {

  String trainSetPath = "E:\\projects\\fractalitylab\\dataset\\class\\julia";

  @Setup
  public void setup() {}

  @Test
  void shouldIterateOverData() throws IOException {
    AugmentationImageReader reader = new AugmentationImageReader();
    Stream<Path> paths = reader.readImagePathsAsStream(trainSetPath);
    paths.parallel().forEach(System.out::println); // 2.38 Mb ram used for 10000 4k Julia images
  }

  @Test
  void shouldIterateOverData2() throws IOException {
    AugmentationImageReader reader = new AugmentationImageReader();
    reader.readImagePathsAsStream2(trainSetPath);
  }
}
