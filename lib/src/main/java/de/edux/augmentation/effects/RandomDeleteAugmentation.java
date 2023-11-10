package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Random;

/**
 * This class provides an augmentation that randomly deletes a portion of the image. This technique
 * is known as CutOut and is used to simulate occlusion and improve robustness and generalization of
 * machine learning models. It encourages the model to focus on the entirety of the object rather
 * than relying on a specific section.
 */
public class RandomDeleteAugmentation extends AbstractAugmentation {

  private final int numberOfCuts;
  private final int cutWidth;
  private final int cutHeight;
  private final Random random;

  /**
   * Constructs a RandomDeleteAugmentation with specified parameters.
   *
   * @param numberOfCuts The number of rectangular areas to delete.
   * @param cutWidth The width of each rectangular cut.
   * @param cutHeight The height of each rectangular cut.
   */
  public RandomDeleteAugmentation(int numberOfCuts, int cutWidth, int cutHeight) {
    this.numberOfCuts = numberOfCuts;
    this.cutWidth = cutWidth;
    this.cutHeight = cutHeight;
    this.random = new Random();
  }

  /**
   * Applies the random delete operation to the given image.
   *
   * @param image The BufferedImage to be augmented.
   * @return A new BufferedImage object with random sections deleted.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    BufferedImage augmentedImage =
        new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
    Graphics2D g2d = augmentedImage.createGraphics();

    g2d.drawImage(image, 0, 0, null);

    for (int i = 0; i < numberOfCuts; i++) {
      int x = random.nextInt(image.getWidth());
      int y = random.nextInt(image.getHeight());
      g2d.setColor(Color.BLACK);
      g2d.fillRect(x, y, cutWidth, cutHeight);
    }

    g2d.dispose();
    return augmentedImage;
  }
}
