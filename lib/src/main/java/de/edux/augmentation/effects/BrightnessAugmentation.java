package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;
import java.awt.image.BufferedImage;

/**
 * This augmentation technique adjusts an image's brightness value by a specified amount
 */
public class BrightnessAugmentation extends AbstractAugmentation {


    private final double brightnessFactor;

    /**
     * Initialize the class with a double value between -1 and 1 indicating the brightness factor to be applied
     * A value greater than 0 will increase brightness
     * A value lower than 0 will decrease brightness
     *
     * @param brightness the brightness factor
     * @throws IllegalArgumentException when the brightness value is not between -1 and 1
     */
    public BrightnessAugmentation(double brightness) throws IllegalArgumentException{
        if (brightness < -1 || brightness > 1)
            throw new IllegalArgumentException("Brightness value must be between -1 and 1");

        this.brightnessFactor = brightness;
    }

    /**
     * Loop through each image pixel and multiply the pixel value by the brightness value

     * @param image the image to augment.
     * @return the image with brightness augmentation applied to it
     */
    @Override
    public BufferedImage apply(BufferedImage image) {
        BufferedImage augmentedImage =
                new BufferedImage(image.getWidth(), image.getHeight(), image.getType());

        boolean hasAlphaChannel = image.getColorModel().hasAlpha();

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);

                int alpha = hasAlphaChannel ? (rgb >> 24) & 0xff : 0xff;
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                int newR = clip((int) (red * (1 + brightnessFactor)));
                int newG = clip((int) (green * (1 + brightnessFactor)));
                int newB = clip((int) (blue * (1 + brightnessFactor)));

                int newRgb = (alpha << 24) | (newR << 16) | (newG << 8) | newB;
                augmentedImage.setRGB(x, y, newRgb);
            }
        }

        return augmentedImage;
    }


    private int clip(int pixelValue){
        return Math.min(255,Math.max(0,pixelValue));
    }
}