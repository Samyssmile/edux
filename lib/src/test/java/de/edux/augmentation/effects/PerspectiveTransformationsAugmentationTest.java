package de.edux.augmentation.effects;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.augmentation.AugmentationBuilder;
import de.edux.augmentation.AugmentationSequence;
import de.edux.augmentation.AugmentationTestUtils;
import de.edux.augmentation.effects.geomentry.Perspective;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class PerspectiveTransformationsAugmentationTest {

  private AugmentationBuilder augmentationBuilder;
  private Path benchmarkDataDir;

  @BeforeEach
  void setUp() throws IOException {}

  @Test
  void shouldApplyAugmentationSequenceOnSingleImage() throws IOException, InterruptedException {
    var image = AugmentationTestUtils.loadTestImage("augmentation/edux-original_3.png");

    double cosAngle = Math.cos(Math.toRadians(30)); // 30 Grad Rotation
    double sinAngle = Math.sin(Math.toRadians(30));
    double[] coefficients2 = {cosAngle, -sinAngle, 0, sinAngle, cosAngle};

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(3840, 2160, ResizeQuality.QUALITY))
            .addAugmentation(
                new PerspectiveTransformationsAugmentation(Perspective.SQUEEZE_VERTICAL))
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

    AugmentationTestUtils.openImageInPreview(image);
    AugmentationTestUtils.openImageInPreview(augmentedImage);
  }
}
