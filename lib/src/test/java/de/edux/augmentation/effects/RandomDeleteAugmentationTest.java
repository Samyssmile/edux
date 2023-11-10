package de.edux.augmentation.effects;

import static de.edux.augmentation.AugmentationTestUtils.loadTestImage;
import static de.edux.augmentation.AugmentationTestUtils.openImageInPreview;
import static org.junit.jupiter.api.Assertions.*;

import de.edux.augmentation.AugmentationBuilder;
import de.edux.augmentation.AugmentationSequence;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

class RandomDeleteAugmentationTest {

  @Test
  void shouldRandomDelete() throws IOException, InterruptedException {
    var image = loadTestImage("augmentation/neo-tokyo.png");
    int originalWidth = image.getWidth();
    int originalHeight = image.getHeight();

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder().addAugmentation(new RandomDeleteAugmentation(5, 40, 80)).build();

    BufferedImage augmentedImage = augmentationSequence.applyTo(image);

    int[] originalPixels =
        image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());
    int[] augmentedPixels =
        augmentedImage.getRGB(
            0,
            0,
            augmentedImage.getWidth(),
            augmentedImage.getHeight(),
            null,
            0,
            augmentedImage.getWidth());

    assertNotNull(augmentedImage, "Augmented image should not be null.");

    assertEquals(
        originalWidth,
        augmentedImage.getWidth(),
        "Augmented image width should match the specified width.");
    assertEquals(
        originalHeight,
        augmentedImage.getHeight(),
        "Augmented image height should match the specified height.");
    assertFalse(
        Arrays.equals(originalPixels, augmentedPixels),
        "The augmented image should differ from the original.");

    Path outputPath = Paths.get("augmented.png");
    ImageIO.write(augmentedImage, "png", outputPath.toFile());

    assertTrue(Files.exists(outputPath), "Output image file should exist.");
    assertTrue(Files.size(outputPath) > 0, "Output image file should not be empty.");

    openImageInPreview(image);
    openImageInPreview(augmentedImage);
  }
}
