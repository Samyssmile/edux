package de.edux.augmentation;

import java.awt.image.BufferedImage;

public interface AugmentationSequence {

  BufferedImage applyTo(BufferedImage image);
}
