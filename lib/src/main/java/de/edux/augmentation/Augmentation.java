package de.edux.augmentation;

import java.awt.image.BufferedImage;
import java.util.List;

/** Interface for an image augmentation sequence. */
public interface Augmentation {

  /**
   * Applies the series of augmentations defined in the sequence to an image.
   *
   * @param image The image to be augmented.
   * @return The augmented image.
   */
  BufferedImage applyTo(BufferedImage image);

  /**
   * Applies the series of augmentations defined in the sequence to a list of images.
   *
   * @param images The list of images to be augmented.
   * @return The list of augmented images.
   */
  List<BufferedImage> applyToAll(List<BufferedImage> images);
}
