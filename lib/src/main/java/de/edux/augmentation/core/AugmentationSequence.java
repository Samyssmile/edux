package de.edux.augmentation.core;

import java.awt.image.BufferedImage;

public interface AugmentationSequence {

  BufferedImage applyTo(BufferedImage image);
}
