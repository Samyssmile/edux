package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;

/** This class provides an augmentation that applies a perspective transformation to an image. */
public class PerspectiveTransformationsAugmentation extends AbstractAugmentation {

  private final double[] coefficients;

  /**
   * Initializes a new instance of the PerspectiveTransformationsAugmentation class with specified
   * coefficients for the transformation.
   *
   * @param coefficients An array of eight coefficients for the perspective transformation matrix.
   */
  public PerspectiveTransformationsAugmentation(double[] coefficients) {

    this.coefficients = coefficients;
  }

  /**
   * Applies a perspective transformation to the provided image.
   *
   * @param image The BufferedImage to which the perspective transformation will be applied.
   * @return A new BufferedImage with the perspective transformation applied.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    int width = image.getWidth();
    int height = image.getHeight();
    BufferedImage transformedImage = new BufferedImage(width, height, image.getType());

    AffineTransform transform = new AffineTransform(coefficients);

    AffineTransformOp op = new AffineTransformOp(transform, AffineTransformOp.TYPE_BILINEAR);
    Graphics2D g2d = transformedImage.createGraphics();
    g2d.drawImage(image, op, 0, 0);
    g2d.dispose();

    return transformedImage;
  }
}
