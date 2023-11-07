package de.edux.augmentation;

import java.awt.image.BufferedImage;

/**
 * Abstract class for augmentation operations. All specific augmentation classes should extend this.
 */
public abstract class AbstractAugmentation {

  /**
   * Applies the augmentation to an image.
   *
   * @param image The image to augment.
   * @return The augmented image.
   */
  public abstract BufferedImage apply(BufferedImage image);
}
