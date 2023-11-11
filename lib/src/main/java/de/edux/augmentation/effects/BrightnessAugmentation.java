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

        for (int y = 0; y < image.getHeight(); y++){
            for (int x = 0; x < image.getWidth(); x++){
                int rgb = image.getRGB(x,y);

                int red   = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue  = rgb & 0xFF;
                int alpha = (rgb >> 24) & 0xff;

                int newR = clip((int) (red   + 255f * brightnessFactor));
                int newG = clip((int) (green + 255f * brightnessFactor));
                int newB = clip((int) (blue  + 255f * brightnessFactor));
                int newRgb = (alpha << 24) | (newR << 16) | (newG << 8) | newB;
                augmentedImage.setRGB(x,y,newRgb);
            }
        }

        return augmentedImage;
    }


    /**
     * Limits the pixel value to a number between 0 and 255
     * if pixelValue is less than 0, returns 0
     * if pixelValue is greater than 255, returns 255
     * else returns the original pixelValue
     */
    private int clip(int pixelValue){
        return Math.min(255,Math.max(0,pixelValue));
    }
}