package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;
import java.awt.*;
import java.awt.image.BufferedImage;

public class ResizeAugmentation extends AbstractAugmentation {

  private final int targetHeight;
  private final int targetWidth;

  private final ResizeQuality resizeQuality;

  public ResizeAugmentation(int targetWidth, int targetHeight) {
    this.targetWidth = targetWidth;
    this.targetHeight = targetHeight;
    this.resizeQuality = ResizeQuality.BALANCED;
  }

  public ResizeAugmentation(int targetWidth, int targetHeight, ResizeQuality resizeQuality) {
    this.targetWidth = targetWidth;
    this.targetHeight = targetHeight;
    this.resizeQuality = resizeQuality;
  }

  @Override
  public synchronized BufferedImage apply(BufferedImage image) {
    BufferedImage result = new BufferedImage(targetWidth, targetHeight, image.getType());
    Graphics2D graphics2D = result.createGraphics();

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
