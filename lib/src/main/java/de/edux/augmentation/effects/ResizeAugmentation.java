package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Provides functionality to resize images to a specified width and height. This class allows for
 * the selection of different resizing qualities from high quality to fast but less refined
 * resizing.
 */
public class ResizeAugmentation extends AbstractAugmentation {

  private final boolean useScaleFactor;
  private final ResizeQuality resizeQuality;
  private int targetHeight;
  private int targetWidth;
  private double scaleFactor;

  /**
   * Creates a ResizeAugmentation instance with the specified target width and height. The resizing
   * quality is set to BALANCED by default.
   *
   * @param targetWidth The desired width of the image after resizing.
   * @param targetHeight The desired height of the image after resizing.
   */
  public ResizeAugmentation(int targetWidth, int targetHeight) {
    this.targetWidth = targetWidth;
    this.targetHeight = targetHeight;
    this.resizeQuality = ResizeQuality.BALANCED;
    this.useScaleFactor = false;
  }

  /**
   * Creates a ResizeAugmentation instance with the specified target width, height, and resizing
   * quality.
   *
   * @param targetWidth The desired width of the image after resizing.
   * @param targetHeight The desired height of the image after resizing.
   * @param resizeQuality The quality of the resize process.
   */
  public ResizeAugmentation(int targetWidth, int targetHeight, ResizeQuality resizeQuality) {
    this.targetWidth = targetWidth;
    this.targetHeight = targetHeight;
    this.resizeQuality = resizeQuality;
    this.useScaleFactor = false;
  }

  /**
   * Creates a ResizeAugmentation instance with a specified scale factor which ensures that the
   * image maintains its aspect ratio after the transformation. A scaleFactor between 0 and 1 will
   * decrease the image size whereas a value greater than 1 will increase the image size.
   *
   * @param scaleFactor The multiplier for the width and height of the image.
   * @throws IllegalArgumentException if scaleFactor is a negative number
   */
  public ResizeAugmentation(double scaleFactor) throws IllegalArgumentException {
    this(scaleFactor, ResizeQuality.BALANCED);
  }

  /**
   * Creates a ResizeAugmentation instance with the specified scaleFactor and resizing quality. A
   * value between 0 and 1 will decrease the image size whereas a value greater than 1 will increase
   * the image size.
   *
   * @param scaleFactor The multiplier for the width and height of the image.
   * @param resizeQuality The quality of the resize process.
   * @throws IllegalArgumentException if scaleFActor is a negative number
   */
  public ResizeAugmentation(double scaleFactor, ResizeQuality resizeQuality)
      throws IllegalArgumentException {
    if (scaleFactor <= 0)
      throw new IllegalArgumentException("Scale factor must be a positive number.");
    this.scaleFactor = scaleFactor;
    this.useScaleFactor = true;
    this.resizeQuality = resizeQuality;
  }

  /**
   * Resizes the given image to the target width and height which are either specified during the
   * instantiation or computed based on the scaleFactor. This method applies rendering hints based
   * on the selected resize quality for the output image.
   *
   * @param image The BufferedImage to resize.
   * @return A new BufferedImage object of the specified target width and height.
   */
  @Override
  public synchronized BufferedImage apply(BufferedImage image) {
    if (useScaleFactor) {
      targetHeight = (int) (image.getHeight() * scaleFactor);
      targetWidth = (int) (image.getWidth() * scaleFactor);
    }
    return resize(image);
  }

  protected BufferedImage resize(BufferedImage image) {
    BufferedImage result = new BufferedImage(targetWidth, targetHeight, image.getType());
    Graphics2D graphics2D = result.createGraphics();
    // Apply rendering hints based on the chosen quality of resizing
    switch (resizeQuality) {
      case QUALITY -> graphics2D.setRenderingHint(
          RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
      case BALANCED -> graphics2D.setRenderingHint(
          RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
      case FAST -> graphics2D.setRenderingHint(
          RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
    }

    graphics2D.drawImage(image, 0, 0, targetWidth, targetHeight, null);
    graphics2D.dispose();
    return result;
  }
}
