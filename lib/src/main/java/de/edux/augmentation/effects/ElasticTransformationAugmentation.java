package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.image.BufferedImage;
import java.util.Random;

/** This class provides an augmentation that applies elastic transformations to an image. */
public class ElasticTransformationAugmentation extends AbstractAugmentation {

  private final double alpha;
  private final double sigma;

  /**
   * Initializes a new instance of the ElasticTransformationAugmentation class with specified
   * parameters.
   *
   * @param alpha Controls the intensity of the deformation. Start with a low value and increase it
   * @param sigma Controls the smoothness of the deformation. Start with a low value and increase it
   */
  public ElasticTransformationAugmentation(double alpha, double sigma) {
    this.alpha = alpha;
    this.sigma = sigma;
  }

  /**
   * Applies an elastic transformation to the provided image.
   *
   * @param image The BufferedImage to which the elastic transformation will be applied.
   * @return A new BufferedImage with the elastic transformation applied.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    int width = image.getWidth();
    int height = image.getHeight();
    BufferedImage transformedImage = new BufferedImage(width, height, image.getType());

    // Generate displacement fields
    Random random = new Random();
    double[] dx = new double[width * height];
    double[] dy = new double[width * height];
    for (int i = 0; i < dx.length; i++) {
      dx[i] = random.nextGaussian() * sigma;
      dy[i] = random.nextGaussian() * sigma;
    }

    // Apply displacement fields to the image
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int newX = (int) (x + dx[y * width + x] * alpha);
        int newY = (int) (y + dy[y * width + x] * alpha);

        // Bound checking
        newX = Math.min(Math.max(newX, 0), width - 1);
        newY = Math.min(Math.max(newY, 0), height - 1);

        transformedImage.setRGB(x, y, image.getRGB(newX, newY));
      }
    }

    return transformedImage;
  }
}
