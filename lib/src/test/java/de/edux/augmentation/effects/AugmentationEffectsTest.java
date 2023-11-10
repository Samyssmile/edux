package de.edux.augmentation.effects;

import static de.edux.augmentation.AugmentationTestUtils.loadTestImage;
import static de.edux.augmentation.AugmentationTestUtils.openImageInPreview;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

public class AugmentationEffectsTest {

  private AugmentationBuilder augmentationBuilder;
  private Path benchmarkDataDir;

  @Test
  void shouldApplyAugmentationSequenceOnSingleImage() throws IOException, InterruptedException {
    var image = loadTestImage("augmentation/edux-original_3.png");

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(3840, 2160, ResizeQuality.QUALITY))
            .addAugmentation(new MonochromeAugmentation())
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
    openImageInPreview(augmentedImage);
  }

  @Test
  void shouldLookLikeRetroPhoto() throws IOException, InterruptedException {
    var originalImage = loadTestImage("augmentation/human-realistic.png");
    int width = originalImage.getWidth();
    int height = originalImage.getHeight();

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(width * 2, height * 2, ResizeQuality.QUALITY))
            .addAugmentation(new ContrastAugmentation(0.6f))
            .addAugmentation(new NoiseInjectionAugmentation(40))
            .addAugmentation(new MonochromeAugmentation())
            .build();

    BufferedImage augmentedImage = augmentationSequence.applyTo(originalImage);

    assertNotNull(augmentedImage, "Augmented originalImage should not be null.");

    assertEquals(
        width * 2,
        augmentedImage.getWidth(),
        "Augmented originalImage width should match the specified width.");
    assertEquals(
        height * 2,
        augmentedImage.getHeight(),
        "Augmented originalImage height should match the specified height.");

    Path outputPath = Paths.get("augmented.png");
    ImageIO.write(augmentedImage, "png", outputPath.toFile());

    assertTrue(Files.exists(outputPath), "Output originalImage file should exist.");
    assertTrue(Files.size(outputPath) > 0, "Output originalImage file should not be empty.");

    // comment this to disable opening the originalImage in the default originalImage viewer
    openImageInPreview(originalImage);
    openImageInPreview(augmentedImage);
  }

  @Test
  void shouldApplyBlurAugmentation() throws IOException, InterruptedException {
    var originalImage = loadTestImage("augmentation/fireworks.png");
    int width = originalImage.getWidth();
    int height = originalImage.getHeight();

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder().addAugmentation(new BlurAugmentation(20)).build();

    BufferedImage augmentedImage = augmentationSequence.applyTo(originalImage);

    assertNotNull(augmentedImage, "Augmented originalImage should not be null.");

    assertEquals(
        width,
        augmentedImage.getWidth(),
        "Augmented originalImage width should match the specified width.");
    assertEquals(
        height,
        augmentedImage.getHeight(),
        "Augmented originalImage height should match the specified height.");

    Path outputPath = Paths.get("augmented.png");
    ImageIO.write(augmentedImage, "png", outputPath.toFile());

    assertTrue(Files.exists(outputPath), "Output originalImage file should exist.");
    assertTrue(Files.size(outputPath) > 0, "Output originalImage file should not be empty.");

    openImageInPreview(originalImage);
    openImageInPreview(augmentedImage);
  }

  @Test
  void shouldApplyCroppingAugmentation() throws IOException, InterruptedException {
    var image = loadTestImage("augmentation/edux-original_3.png");

    int width = image.getWidth();
    int height = image.getHeight();

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder().addAugmentation(new CroppingAugmentation(0.5f)).build();

    BufferedImage augmentedImage = augmentationSequence.applyTo(image);

    assertNotNull(augmentedImage, "Augmented image should not be null.");

    assertEquals(
        width / 2,
        augmentedImage.getWidth(),
        "Augmented image width should match the specified width.");
    assertEquals(
        height / 2,
        augmentedImage.getHeight(),
        "Augmented image height should match the specified height.");

    Path outputPath = Paths.get("augmented.png");
    ImageIO.write(augmentedImage, "png", outputPath.toFile());

    assertTrue(Files.exists(outputPath), "Output image file should exist.");
    assertTrue(Files.size(outputPath) > 0, "Output image file should not be empty.");

    openImageInPreview(augmentedImage);
  }
}
