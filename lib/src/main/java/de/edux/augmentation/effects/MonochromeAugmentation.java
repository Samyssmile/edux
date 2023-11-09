package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.image.BufferedImage;

/** Applies a monochrome filter to the image, converting it to grayscale. */
public class MonochromeAugmentation extends AbstractAugmentation {

  @Override
  public BufferedImage apply(BufferedImage image) {
    BufferedImage monochromeImage =
        new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
    // Creating graphics for the BufferedImage allows us to draw the original image onto the new
    // one.
    var graphics = monochromeImage.getGraphics();
    graphics.drawImage(image, 0, 0, null);
    graphics.dispose(); // Always dispose of the graphics object when done with it.
    return monochromeImage;
  }
}
