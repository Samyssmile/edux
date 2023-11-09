package de.edux.augmentation;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * Abstract class for augmentation operations. All specific augmentation classes should extend this.
 */
public abstract class AbstractAugmentation {

  public static BufferedImage readImage(Path imagePath) throws IOException {
    return ImageIO.read(imagePath.toFile());
  }

  /**
   * Applies the augmentation to an image.
   *
   * @param image The image to augment.
   * @return The augmented image.
   */
  public abstract BufferedImage apply(BufferedImage image);
}
