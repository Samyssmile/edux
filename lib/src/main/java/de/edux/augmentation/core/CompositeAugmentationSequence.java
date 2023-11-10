package de.edux.augmentation.core;

import java.awt.image.BufferedImage;
import java.util.List;

/** Composite augmentation that applies a sequence of augmentations. */
public class CompositeAugmentationSequence implements AugmentationSequence {
  private final List<AbstractAugmentation> augmentations;

  public CompositeAugmentationSequence(List<AbstractAugmentation> augmentations) {
    this.augmentations = augmentations;
  }

  @Override
  public BufferedImage applyTo(BufferedImage image) {
    BufferedImage currentImage = image;
    for (AbstractAugmentation augmentation : augmentations) {
      currentImage = augmentation.apply(currentImage);
    }
    return currentImage;
  }
}
