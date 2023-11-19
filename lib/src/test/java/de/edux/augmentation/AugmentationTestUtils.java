package de.edux.augmentation;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import javax.imageio.ImageIO;

public class AugmentationTestUtils {
  private static final boolean OPEN_IMAGES_IN_PREVIEW = false;

  public static void openImageInPreview(BufferedImage augmentedImage) throws InterruptedException {
    if (OPEN_IMAGES_IN_PREVIEW) {
      Path tempFile = null;
      try {

        if (Desktop.isDesktopSupported()) {
          tempFile = Files.createTempFile(UUID.randomUUID().toString(), ".png");
          ImageIO.write(augmentedImage, "png", tempFile.toFile());

          Desktop.getDesktop().open(tempFile.toFile());
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  public static BufferedImage loadTestImage(String path) throws IOException {
    var resourcePath = path;
    var imageStream =
        AugmentationTestUtils.class.getClassLoader().getResourceAsStream(resourcePath);
    if (imageStream == null) {
      throw new IOException("Cannot find resource: " + resourcePath);
    }
    return ImageIO.read(imageStream);
  }
}
