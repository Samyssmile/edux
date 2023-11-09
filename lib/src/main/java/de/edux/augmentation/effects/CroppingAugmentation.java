package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.image.BufferedImage;

/** This class provides an augmentation that crops an image based on a specified factor. */
public class CroppingAugmentation extends AbstractAugmentation {

  private final float cropFactor;

  /**
   * Constructs a CroppingAugmentation with a specified crop factor. The crop factor should be
   * between 0 and 1, where 1 would result in no crop and 0.5 would crop half of the width and
   * height.
   *
   * @param cropFactor The factor by which the image should be cropped.
   */
  public CroppingAugmentation(float cropFactor) {
    if (cropFactor <= 0 || cropFactor > 1) {
      throw new IllegalArgumentException("Crop factor must be between 0 and 1.");
    }
    this.cropFactor = cropFactor;
  }

  /**
   * Applies a cropping augmentation to the provided image.
   *
   * @param image The BufferedImage to which the cropping augmentation will be applied.
   * @return A new BufferedImage with the cropping augmentation applied.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    int width = image.getWidth();
    int height = image.getHeight();
    int newWidth = (int) (width * cropFactor);
    int newHeight = (int) (height * cropFactor);

    int x0 = (width - newWidth) / 2;
    int y0 = (height - newHeight) / 2;

    return image.getSubimage(x0, y0, newWidth, newHeight);
  }
}
