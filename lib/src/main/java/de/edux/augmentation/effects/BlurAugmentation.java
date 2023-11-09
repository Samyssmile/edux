package de.edux.augmentation.effects;

import de.edux.augmentation.AbstractAugmentation;

import java.awt.image.BufferedImage;
import java.awt.image.Kernel;
import java.awt.image.ConvolveOp;

/** This class provides an augmentation that applies a Gaussian blur to an image. */
public class BlurAugmentation extends AbstractAugmentation {

  private final float radius;

  /**
   * Initializes a new instance of the BlurAugmentation class with the specified blur radius.
   *
   * @param radius The radius of the Gaussian blur. Higher values result in a more pronounced blur
   *     effect.
   */
  public BlurAugmentation(float radius) {
    this.radius = radius;
  }

  /**
   * Applies a Gaussian blur augmentation to the provided image.
   *
   * @param image The BufferedImage to which the blur augmentation will be applied.
   * @return A new BufferedImage with the blur augmentation applied.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    int radius = (int) this.radius;
    int size = radius * 2 + 1;
    float[] data = new float[size * size];

    float sigma = radius / 3.0f;
    float twoSigmaSquare = 2.0f * sigma * sigma;
    float sigmaRoot = (float) Math.sqrt(twoSigmaSquare * Math.PI);
    float total = 0.0f;

    for (int y = -radius; y <= radius; y++) {
      for (int x = -radius; x <= radius; x++) {
        float distance = x * x + y * y;
        int index = (y + radius) * size + x + radius;
        data[index] = (float) Math.exp(-distance / twoSigmaSquare) / sigmaRoot;
        total += data[index];
      }
    }

    for (int i = 0; i < data.length; i++) {
      data[i] /= total;
    }

    Kernel kernel = new Kernel(size, size, data);
    ConvolveOp op = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);
    return op.filter(image, null);
  }
}
