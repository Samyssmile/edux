package de.example.augmentation;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import de.edux.augmentation.effects.ColorEqualizationAugmentation;
import de.edux.augmentation.effects.ResizeAugmentation;
import java.awt.*;
import java.io.File;
import java.io.IOException;

public class MultiImageAugmentation {
  private static final String IMAGE_DIR =
      "example"
          + File.separator
          + "src"
          + File.separator
          + "main"
          + File.separator
          + "resources"
          + File.separator
          + "images"
          + File.separator
          + "small-julia";

  private static final Integer TARGET_WIDTH = 250;
  private static final Integer TARGET_HEIGHT = 250;
  private static final int TWO_CPU_WORKERS = 2;

  public static void main(String[] args) throws IOException, InterruptedException {
    String projectRootPath = new File("").getAbsolutePath();
    String imageDirPath = projectRootPath + File.separator + IMAGE_DIR;
    String outputFolder = imageDirPath + File.separator + "augmented";

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(TARGET_WIDTH, TARGET_HEIGHT))
            .addAugmentation(new ColorEqualizationAugmentation())
            .build()
            .run(imageDirPath, TWO_CPU_WORKERS, outputFolder);

    openFolder(outputFolder);
  }

  private static void openFolder(String outputFolder) throws IOException {
    File augmentedImagesDir = new File(outputFolder);
    if (Desktop.isDesktopSupported()) {
      Desktop.getDesktop().open(augmentedImagesDir);
    } else {
      System.out.println(
          "No Desktop support. Please open the augmented images directory manually.");
    }
  }
}
