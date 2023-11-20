package de.edux.augmentation.core;

import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Defines a sequence of image augmentation operations.
 *
 * <p>This interface represents a sequence of augmentation operations that can be applied to images.
 * It allows for both the application of augmentations to a single image and the execution of the
 * augmentation sequence on a batch of images located in a directory.
 */
public interface AugmentationSequence {

  /**
   * Applies the augmentation sequence to a single image.
   *
   * <p>This method takes a {@link BufferedImage} and applies the defined sequence of augmentation
   * operations to it, returning the augmented image.
   *
   * @param image The image to which the augmentation sequence will be applied.
   * @return The augmented image as a {@link BufferedImage}.
   */
  BufferedImage applyTo(BufferedImage image);

  /**
   * Executes the augmentation sequence on a set of images in a specified directory.
   *
   * <p>This method applies the augmentation sequence to each image in the specified directory,
   * using a specified number of workers for parallel processing. The augmented images are then
   * saved to the given output path.
   *
   * @param directoryPath The path of the directory containing the images to augment.
   * @param numberOfWorkers The number of worker threads to use for parallel processing.
   * @param outputPath The path where the augmented images will be saved.
   * @return The {@link AugmentationSequence} instance for method chaining or further operations.
   * @throws IOException If an I/O error occurs during the reading or writing of images.
   * @throws InterruptedException If the thread execution is interrupted during processing.
   */
  AugmentationSequence run(String directoryPath, int numberOfWorkers, String outputPath)
      throws IOException, InterruptedException;
}
