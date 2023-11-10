package de.edux.augmentation.effects;

import static de.edux.augmentation.AugmentationTestUtils.loadTestImage;
import static de.edux.augmentation.AugmentationTestUtils.openImageInPreview;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

public class ElasticTransformationTest {

  @Test
  void shouldApplyAugmentationSequenceOnSingleImage() throws IOException, InterruptedException {
    var image = loadTestImage("augmentation" + File.separator + "edux-original_3.png");

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

    openImageInPreview(image);
    openImageInPreview(augmentedImage);
  }
}
