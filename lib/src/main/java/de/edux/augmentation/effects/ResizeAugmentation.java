package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Provides functionality to resize images to a specified width and height. This class allows for
 * the selection of different resizing qualities from high quality to fast but less refined
 * resizing.
 */
public class ResizeAugmentation extends AbstractAugmentation {

  private final int targetHeight;
  private final int targetWidth;
  private final ResizeQuality resizeQuality;

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
  }

  /**
   * Resizes the given image to the target width and height specified during the instantiation. This
   * method applies rendering hints based on the selected resize quality for the output image.
   *
   * @param image The BufferedImage to resize.
   * @return A new BufferedImage object of the specified target width and height.
   */
  @Override
  public synchronized BufferedImage apply(BufferedImage image) {
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
