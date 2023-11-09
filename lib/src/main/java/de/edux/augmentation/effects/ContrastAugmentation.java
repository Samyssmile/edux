package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.Color;
import java.awt.image.BufferedImage;

/** This class enhances the contrast of an image. */
public class ContrastAugmentation extends AbstractAugmentation {

  private final float scaleFactor;

  /**
   * Initializes a new instance of the ContrastAugmentation class with the specified contrast
   * factor.
   *
   * @param contrast A value between 0 and 1, where higher values indicate a higher level of
   *     contrast.
   */
  public ContrastAugmentation(float contrast) {
    if (contrast < 0 || contrast > 1) {
      throw new IllegalArgumentException("Contrast must be between 0 and 1.");
    }
    this.scaleFactor = 1 + (contrast - 0.5f) * 2;
  }

  /**
   * Applies contrast enhancement to the provided image.
   *
   * @param image The BufferedImage to which the contrast enhancement will be applied.
   * @return A new BufferedImage with the contrast enhancement applied.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    BufferedImage contrastedImage =
        new BufferedImage(image.getWidth(), image.getHeight(), image.getType());

    for (int y = 0; y < image.getHeight(); y++) {
      for (int x = 0; x < image.getWidth(); x++) {
        int rgba = image.getRGB(x, y);
        Color col = new Color(rgba, true);
        int r = (int) bound((int) (col.getRed() * scaleFactor));
        int g = (int) bound((int) (col.getGreen() * scaleFactor));
        int b = (int) bound((int) (col.getBlue() * scaleFactor));

        Color newColor = new Color(bound(r), bound(g), bound(b), col.getAlpha());
        contrastedImage.setRGB(x, y, newColor.getRGB());
      }
    }

    return contrastedImage;
  }

  private int bound(int colorValue) {
    return Math.min(Math.max(colorValue, 0), 255);
  }
}
