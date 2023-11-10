package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

/**
 * This class implements an augmentation that applies color equalization to an image. It enhances
 * the contrast of the image by equalizing the histogram of each color channel.
 */
public class ColorEqualizationAugmentation extends AbstractAugmentation {

  /**
   * Applies color equalization to the provided image.
   *
   * @param image The BufferedImage to be augmented.
   * @return The augmented BufferedImage with equalized color channels.
   */
  @Override
  public BufferedImage apply(BufferedImage image) {
    BufferedImage equalizedImage =
        new BufferedImage(image.getWidth(), image.getHeight(), image.getType());

    WritableRaster sourceRaster = image.getRaster();
    WritableRaster equalizedRaster = equalizedImage.getRaster();

    for (int b = 0; b < sourceRaster.getNumBands(); b++) {
      equalizeHistogram(sourceRaster, equalizedRaster, b);
    }

    return equalizedImage;
  }

  private void equalizeHistogram(
      WritableRaster sourceRaster, WritableRaster equalizedRaster, int band) {
    int w = sourceRaster.getWidth();
    int h = sourceRaster.getHeight();
    int[] histogram = new int[256];
    float[] cdf = new float[256];

    // Calculate the histogram
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        histogram[sourceRaster.getSample(x, y, band) & 0xFF]++;
      }
    }

    int total = w * h;
    float sum = 0f;
    for (int i = 0; i < histogram.length; i++) {
      sum += (float) histogram[i] / total;
      cdf[i] = sum;
    }

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int value = sourceRaster.getSample(x, y, band) & 0xFF;
        int newValue = Math.round(cdf[value] * 255f);
        equalizedRaster.setSample(x, y, band, newValue);
      }
    }
  }
}
