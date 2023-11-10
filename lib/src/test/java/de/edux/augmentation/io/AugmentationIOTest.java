package de.edux.augmentation.io;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import de.edux.augmentation.effects.BlurAugmentation;
import de.edux.augmentation.effects.ColorEqualizationAugmentation;
import de.edux.augmentation.effects.RandomDeleteAugmentation;
import de.edux.augmentation.effects.ResizeAugmentation;
import java.io.IOException;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.openjdk.jmh.annotations.Setup;

@Disabled
public class AugmentationIOTest {

  private final int numberOfWorkers = 6;
  String trainImagesDir = "image/path";
  String outputDir = "output/path";

  @Setup
  public void setup() {}

  @Test
  void shouldIterateOverData() throws IOException, InterruptedException {
    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(250, 250))
            .addAugmentation(new ColorEqualizationAugmentation())
            .addAugmentation(new BlurAugmentation(25))
            .addAugmentation(new RandomDeleteAugmentation(10, 20, 20))
            .build()
            .run(trainImagesDir, numberOfWorkers, outputDir);
  }
}
