package de.edux.augmentation.effects;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.augmentation.AugmentationBuilder;
import de.edux.augmentation.AugmentationSequence;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.UUID;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ElasticTransformationTest {
  private static final boolean OPEN_IMAGES_IN_PREVIEW = false;

  private AugmentationBuilder augmentationBuilder;
  private Path benchmarkDataDir;

  @BeforeEach
  void setUp() throws IOException {}

  @Test
  void shouldApplyAugmentationSequenceOnSingleImage() throws IOException, InterruptedException {
    var image = loadTestImage("augmentation/edux-original_3.png");

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(3840, 2160, ResizeQuality.QUALITY))
            .addAugmentation(new ElasticTransformationAugmentation(5, 2))
            .build();

    BufferedImage augmentedImage = augmentationSequence.applyTo(image);

    assertNotNull(augmentedImage, "Augmented image should not be null.");

    assertEquals(
        3840, augmentedImage.getWidth(), "Augmented image width should match the specified width.");
    assertEquals(
        2160,
        augmentedImage.getHeight(),
        "Augmented image height should match the specified height.");

    Path outputPath = Paths.get("augmented.png");
    ImageIO.write(augmentedImage, "png", outputPath.toFile());

    assertTrue(Files.exists(outputPath), "Output image file should exist.");
    assertTrue(Files.size(outputPath) > 0, "Output image file should not be empty.");

    // comment this to disable opening the image in the default image viewer
    openImageInPreview(image);
    openImageInPreview(augmentedImage);
  }

  private void openImageInPreview(BufferedImage augmentedImage) throws InterruptedException {
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

  private BufferedImage loadTestImage(String path) throws IOException {
    var resourcePath = path;
    var imageStream = this.getClass().getClassLoader().getResourceAsStream(resourcePath);
    if (imageStream == null) {
      throw new IOException("Cannot find resource: " + resourcePath);
    }
    return ImageIO.read(imageStream);
  }
}
