package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;
import java.awt.image.BufferedImage;
import java.util.Random;

public class NoiseInjectionAugmentation extends AbstractAugmentation {

  private final int noiseLevel;

  public NoiseInjectionAugmentation(int noiseLevel) {
    this.noiseLevel = noiseLevel;
  }

  @Override
  public BufferedImage apply(BufferedImage image) {
    Random rand = new Random();

    int width = image.getWidth();
    int height = image.getHeight();

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int p = image.getRGB(x, y);

        int a = (p >> 24) & 0xff;
        int r = (p >> 16) & 0xff;
        int g = (p >> 8) & 0xff;
        int b = p & 0xff;

        int noise = rand.nextInt(noiseLevel * 2 + 1) - noiseLevel;
        r = Math.min(Math.max(r + noise, 0), 255);
        g = Math.min(Math.max(g + noise, 0), 255);
        b = Math.min(Math.max(b + noise, 0), 255);

        p = (a << 24) | (r << 16) | (g << 8) | b;
        image.setRGB(x, y, p);
      }
    }

    return image;
  }
}
