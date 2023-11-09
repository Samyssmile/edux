package de.edux.augmentation.composite;

import de.edux.augmentation.AbstractAugmentation;
import de.edux.augmentation.AugmentationSequence;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

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
