package de.edux.augmentation.effects;

import org.junit.jupiter.api.AfterEach;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

public abstract class AbstractAugmentationTest {
  protected BufferedImage originalImage;
  protected BufferedImage augmentedImage;

  public void shouldHaveSameWidth() {
    assertEquals(
        originalImage.getWidth(),
        augmentedImage.getWidth(),
        "Augmented image width should match the specified width.");
  }

  public void shouldHaveSameHeight() {
    assertEquals(
        originalImage.getHeight(),
        augmentedImage.getHeight(),
        "Augmented image height should match the specified height.");
  }

  public void shouldHaveSamePixels() {
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
    assertFalse(
        Arrays.equals(originalPixels, augmentedPixels),
        "The augmented image should differ from the original.");
  }

  public void shouldNotBeNull() {
    assertNotNull(augmentedImage, "Augmented image should not be null.");
  }

  public void outputFileShouldExistAndNotBeEmpty() throws IOException {
    Path outputPath = Paths.get("augmented.png");
    ImageIO.write(augmentedImage, "png", outputPath.toFile());
    assertTrue(Files.exists(outputPath), "Output image file should exist.");
    assertTrue(Files.size(outputPath) > 0, "Output image file should not be empty.");
  }

  @AfterEach
  public void checkConformity() throws IOException {
    shouldNotBeNull();
    shouldHaveSameWidth();
    shouldHaveSameHeight();
    shouldHaveSamePixels();
    outputFileShouldExistAndNotBeEmpty();
  }
}
