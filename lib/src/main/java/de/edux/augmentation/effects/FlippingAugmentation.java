package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;
import java.awt.image.BufferedImage;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This class provides an augmentation that randomly flips an image either horizontally or
 * vertically.
 */
public class FlippingAugmentation extends AbstractAugmentation {

  private final ThreadLocalRandom random = ThreadLocalRandom.current();

  /**
   * Applies a random flip augmentation to the provided image. The flip is either vertical or
   * horizontal and is chosen at random.
   *
   * @param image The BufferedImage to which the flipping augmentation will be applied.
   * @return A new BufferedImage with the flip augmentation applied.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    int width = image.getWidth();
    int height = image.getHeight();

    BufferedImage flippedImage = new BufferedImage(width, height, image.getType());

    if (random.nextDouble() > 0.5) {
      // Flip the image vertically
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          flippedImage.setRGB(x, (height - 1) - y, image.getRGB(x, y));
        }
      }
    } else {
      // Flip the image horizontally
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          flippedImage.setRGB((width - 1) - x, y, image.getRGB(x, y));
        }
      }
    }

    return flippedImage;
  }
}
