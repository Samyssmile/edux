package de.edux.augmentation.core;

import java.awt.image.BufferedImage;
import java.io.IOException;

public interface AugmentationSequence {

  BufferedImage applyTo(BufferedImage image);

  AugmentationSequence run(String directoryPath, int numberOfWorkers, String outputPath)
      throws IOException, InterruptedException;
}
