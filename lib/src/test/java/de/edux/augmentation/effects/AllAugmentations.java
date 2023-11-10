package de.edux.augmentation.effects;

import static de.edux.augmentation.AugmentationTestUtils.loadTestImage;
import static de.edux.augmentation.AugmentationTestUtils.openImageInPreview;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import de.edux.augmentation.effects.geomentry.Perspective;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

public class AllAugmentations {

  @Test
  void shouldApplyAllAugmentations() throws InterruptedException, IOException {
    var originalImage = loadTestImage("augmentation/national-park.png");
    int width = originalImage.getWidth();
    int height = originalImage.getHeight();

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(width * 2, height * 2, ResizeQuality.QUALITY))
            .addAugmentation(new ContrastAugmentation(0.6f))
            .addAugmentation(new BlurAugmentation(5))
            .addAugmentation(new NoiseInjectionAugmentation(40))
            .addAugmentation(new MonochromeAugmentation())
            .addAugmentation(new FlippingAugmentation())
            .addAugmentation(new ColorEqualizationAugmentation())
            .addAugmentation(new CroppingAugmentation(0.2f))
            .addAugmentation(new ElasticTransformationAugmentation(5, 0.5))
            .addAugmentation(new PerspectiveTransformationsAugmentation(Perspective.RIGHT_TILT))
            .addAugmentation(new RandomDeleteAugmentation(5, 20, 20))
            .build();

    BufferedImage augmentedImage = augmentationSequence.applyTo(originalImage);
    int[] originalPixels =
        originalImage.getRGB(
            0,
            0,
            originalImage.getWidth(),
            originalImage.getHeight(),
            null,
            0,
            originalImage.getWidth());
    int[] augmentedPixels =
        augmentedImage.getRGB(
            0,
            0,
            augmentedImage.getWidth(),
            augmentedImage.getHeight(),
            null,
            0,
            augmentedImage.getWidth());
    assertNotNull(augmentedImage, "Augmented originalImage should not be null.");

    assertEquals(
        515,
        augmentedImage.getWidth(),
        "Augmented originalImage width should match the specified width.");
    assertEquals(
        409,
        augmentedImage.getHeight(),
        "Augmented originalImage height should match the specified height.");

    Path outputPath = Paths.get("augmented.png");
    ImageIO.write(augmentedImage, "png", outputPath.toFile());

    assertTrue(Files.exists(outputPath), "Output originalImage file should exist.");
    assertTrue(Files.size(outputPath) > 0, "Output originalImage file should not be empty.");
    assertFalse(
        Arrays.equals(originalPixels, augmentedPixels),
        "The augmented image should differ from the original.");

    // comment this to disable opening the originalImage in the default originalImage viewer
    openImageInPreview(originalImage);
    openImageInPreview(augmentedImage);
  }
}
