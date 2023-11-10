package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;

/** This class provides an augmentation that applies a Gaussian blur to an image. */
public class BlurAugmentation extends AbstractAugmentation {

  private final float radius;

  /**
   * Initializes a new instance of the BlurAugmentation class with the specified blur radius. The
   * radius determines the size of the area over which the Gaussian blur is applied. A larger radius
   * results in a greater blur effect, where more distant pixels influence each other, leading to a
   * smoother, more blurred image. A smaller radius restricts this effect to closer pixels,
   * resulting in a less blurred image. The radius is not constrained between 0 and 1; it can take
   * on any positive value, with typical values ranging from 1 for a subtle blur, up to 5 or more
   * for a more significant blur effect.
   *
   * @param radius The radius of the Gaussian blur. This is not a percentage but a pixel count that
   *     defines the size of the kernel to be applied. The actual kernel size used in the
   *     convolution will be (radius * 2 + 1) to ensure that it covers both sides of the central
   *     pixel. Typical values range from 1 to 5 or more.
   */
  public BlurAugmentation(float radius) {
    if (radius < 0) {
      throw new IllegalArgumentException("Radius must be greater than 0.");
    }
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
