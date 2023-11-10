package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.image.BufferedImage;
import java.util.Random;

/** Applies an elastic transformation to an image, simulating natural distortions. */
public class ElasticTransformationAugmentation extends AbstractAugmentation {

  private final double alpha;
  private final double sigma;

  /**
   * Constructs an ElasticTransformationAugmentation instance with the given parameters.
   *
   * @param alpha The intensity of the transformation.
   * @param sigma The elasticity coefficient.
   */
  public ElasticTransformationAugmentation(double alpha, double sigma) {
    this.alpha = alpha;
    this.sigma = sigma;
  }

  /**
   * Applies the elastic transformation to the provided image.
   *
   * @param image The BufferedImage to transform.
   * @return A new BufferedImage object with the applied elastic transformation.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    int width = image.getWidth();
    int height = image.getHeight();
    BufferedImage result = new BufferedImage(width, height, image.getType());

    // Generate displacement fields
    double[] dx = generateDisplacementField(width, height, sigma);
    double[] dy = generateDisplacementField(width, height, sigma);

    // Apply the elastic transformation
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int xx = (int) (x + alpha * dx[y * width + x]);
        int yy = (int) (y + alpha * dy[y * width + x]);

        // Boundary condition check
        if (xx < 0) xx = 0;
        if (xx >= width) xx = width - 1;
        if (yy < 0) yy = 0;
        if (yy >= height) yy = height - 1;

        result.setRGB(x, y, image.getRGB(xx, yy));
      }
    }

    return result;
  }

  /**
   * Generates a displacement field for the elastic transformation.
   *
   * @param width The width of the displacement field.
   * @param height The height of the displacement field.
   * @param sigma The elasticity coefficient for the displacement field.
   * @return A displacement field represented as a double array.
   */
  private double[] generateDisplacementField(int width, int height, double sigma) {
    double[] field = new double[width * height];
    Random random = new Random();

    for (int i = 0; i < field.length; i++) {
      field[i] = random.nextGaussian() * sigma;
    }

    // Here you might want to apply a Gaussian blur to the field
    // to ensure smoothness of the displacement.

    return field;
  }
}
