package de.edux.augmentation.effects.geomentry;

/**
 * Defines common perspective transformations that can be applied to images. Each enum constant
 * represents a predefined set of affine transformations that mimic the effect of changing the
 * perspective from which an image is viewed.
 */
public enum Perspective {
  RIGHT_TILT(1.0, 1.0, 0.5, 0.0, -0.25, 0.0),
  LEFT_TILT(1.0, 1.0, -0.5, 0.0, 0.25, 0.0),
  TOP_TILT(1.0, 1.0, 0.0, 0.5, 0.0, -0.25),
  BOTTOM_TILT(1.0, 1.0, 0.0, -0.5, 0.0, 0.25),
  TOP_RIGHT_CORNER_TILT(1.0, 1.0, 0.3, 0.3, -0.15, -0.15),
  TOP_LEFT_CORNER_TILT(1.0, 1.0, -0.3, 0.3, 0.15, -0.15),
  BOTTOM_RIGHT_CORNER_TILT(1.0, 1.0, 0.3, -0.3, -0.15, 0.15),
  BOTTOM_LEFT_CORNER_TILT(1.0, 1.0, -0.3, -0.3, 0.15, 0.15),
  STRETCH_HORIZONTAL(1.2, 1.0, 0.0, 0.0, -0.1, 0.0),
  STRETCH_VERTICAL(1.0, 1.2, 0.0, 0.0, 0.0, -0.1),
  SQUEEZE_HORIZONTAL(0.8, 1.0, 0.0, 0.0, 0.1, 0.0),
  SQUEEZE_VERTICAL(1.0, 0.8, 0.0, 0.0, 0.0, 0.1);

  private final double scaleX;
  private final double scaleY;
  private final double shearX;
  private final double shearY;
  private final double translateX;
  private final double translateY;

  Perspective(
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

  public double getScaleX() {
    return scaleX;
  }

  public double getScaleY() {
    return scaleY;
  }

  public double getShearX() {
    return shearX;
  }

  public double getShearY() {
    return shearY;
  }

  public double getTranslateX() {
    return translateX;
  }

  public double getTranslateY() {
    return translateY;
  }
}
