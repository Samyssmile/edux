package de.edux.augmentation.io;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import de.edux.augmentation.effects.ColorEqualizationAugmentation;
import de.edux.augmentation.processor.ImageAugmentationProcessor;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
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
    try {
      String inputDir = trainSetPath;
      String outputDir = "output";

      AugmentationSequence augmentationSequence =
          new AugmentationBuilder().addAugmentation(new ColorEqualizationAugmentation()).build();

      IAugmentationImageReader reader = new AugmentationImageReader();
      ImageAugmentationProcessor processor =
          new ImageAugmentationProcessor(
              augmentationSequence, reader, Paths.get(inputDir), Paths.get(outputDir));

      processor.processImages();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
