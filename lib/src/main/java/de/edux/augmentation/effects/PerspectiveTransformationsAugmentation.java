package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;
import de.edux.augmentation.effects.geomentry.Perspective;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;

/**
 * Applies a perspective transformation to an image using affine transformations. This class
 * supports custom transformations based on scale, shear, and translation parameters, as well as
 * predefined perspectives provided by the Perspective enum.
 */
public class PerspectiveTransformationsAugmentation extends AbstractAugmentation {

  private double scaleX, scaleY, shearX, shearY, translateX, translateY;

  /**
   * Constructs a PerspectiveTransformationsAugmentation with predefined perspective settings.
   *
   * @param perspective The predefined perspective transformation to apply.
   */
  public PerspectiveTransformationsAugmentation(Perspective perspective) {
    this.scaleX = perspective.getScaleX();
    this.scaleY = perspective.getScaleY();
    this.shearX = perspective.getShearX();
    this.shearY = perspective.getShearY();
    this.translateX = perspective.getTranslateX();
    this.translateY = perspective.getTranslateY();
  }

  /**
   * Constructs a PerspectiveTransformationsAugmentation with custom transformation settings.
   *
   * @param scaleX The factor by which the image is scaled in X direction.
   * @param scaleY The factor by which the image is scaled in Y direction.
   * @param shearX The X coordinate shearing factor.
   * @param shearY The Y coordinate shearing factor.
   * @param translateX The horizontal translation distance.
   * @param translateY The vertical translation distance.
   */
  public PerspectiveTransformationsAugmentation(
      double scaleX,
      double scaleY,
      double shearX,
      double shearY,
      double translateX,
      double translateY) {
    this.scaleX = scaleX;
    this.scaleY = scaleY;
    this.shearX = shearX;
    this.shearY = shearY;
    this.translateX = translateX;
    this.translateY = translateY;
  }

  /**
   * Applies the affine transformation to the provided image.
   *
   * @param image The original image to be transformed.
   * @return A new BufferedImage object with the applied perspective transformation.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    int width = image.getWidth();
    int height = image.getHeight();
    BufferedImage transformedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    Graphics2D g2d = transformedImage.createGraphics();

    AffineTransform transform = new AffineTransform();
    transform.translate(translateX, translateY);
    transform.scale(scaleX, scaleY);
    transform.shear(shearX, shearY);

    g2d.setTransform(transform);
    g2d.drawImage(image, 0, 0, null);
    g2d.dispose();

    return transformedImage;
  }
}
