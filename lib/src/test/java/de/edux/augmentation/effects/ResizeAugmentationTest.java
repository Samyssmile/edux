package de.edux.augmentation.effects;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static de.edux.augmentation.AugmentationTestUtils.openImageInPreview;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.io.IOException;

import static de.edux.augmentation.AugmentationTestUtils.loadTestImage;

public class ResizeAugmentationTest extends AbstractAugmentationTest {

  private double originalAspectRatio;

  @Override
  @AfterEach
  public void checkConformity() throws IOException {
    shouldNotBeNull();
    shouldHaveSamePixels();
    outputFileShouldExistAndNotBeEmpty();
  }

  @AfterEach
  public void shouldMaintainAspectRatio() {
    double newAspectRatio = (double) augmentedImage.getWidth() / augmentedImage.getHeight();
    assertEquals(
        originalAspectRatio,
        newAspectRatio,
        "Augmented image should have the same aspect ratio as the original image");
  }

  @AfterEach
  public void openImagesInPreview() throws InterruptedException {
    openImageInPreview(originalImage);
    openImageInPreview(augmentedImage);
  }

  @Test
  public void shouldIncreaseImageSize() throws IOException {
    originalImage = loadTestImage("augmentation" + File.separator + "cyborg-cyberpunk.png");
    originalAspectRatio = (double) originalImage.getWidth() / originalImage.getHeight();

    double testScaleFactor = 2;
    ResizeAugmentation ResizeAugmentation = new ResizeAugmentation(testScaleFactor);
    augmentedImage = ResizeAugmentation.apply(originalImage);

    assertEquals(augmentedImage.getHeight(), originalImage.getHeight() * testScaleFactor);
    assertEquals(augmentedImage.getWidth(), originalImage.getWidth() * testScaleFactor);
  }

  @Test
  public void shouldDecreaseImageSize() throws IOException {
    originalImage = loadTestImage("augmentation" + File.separator + "neo-tokyo.png");
    originalAspectRatio = (double) originalImage.getWidth() / originalImage.getHeight();

    double testScaleFactor = 0.5;
    ResizeAugmentation ResizeAugmentation = new ResizeAugmentation(testScaleFactor);
    augmentedImage = ResizeAugmentation.apply(originalImage);

    assertEquals(augmentedImage.getHeight(), originalImage.getHeight() * testScaleFactor);
    assertEquals(augmentedImage.getWidth(), originalImage.getWidth() * testScaleFactor);
  }
}
