package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;

import java.awt.Graphics;
import java.awt.image.BufferedImage;

/**
 * Applies Otsu's thresholding method to the given image.
 * Converts the image to grayscale, calculates the histogram, finds the optimal threshold,
 * and applies the threshold to create a binary image.
 */
public class OtsuThresholdingAugmentation extends AbstractAugmentation {
  private static final Integer HISTOGRAM_SIZE = 256;
    private static final int RED_INTENSITY_SHIFT = 16;

    /**
     * Applies Otsu's thresholding method to the given image.
     * @param image the original image to be thresholded
     * @return a binary image where pixels with intensity greater than the threshold are white,
     * and others are black
     */
    @Override
    public BufferedImage apply(BufferedImage image) {
        BufferedImage grayImage = toGrayScale((image));
        int[] histogram = calculateHistogram(grayImage);
        int threshold = findOtsuThreshold(histogram, grayImage.getWidth() * grayImage.getHeight());
        BufferedImage thresholdedImage = new BufferedImage(grayImage.getWidth(), grayImage.getHeight(), BufferedImage.TYPE_BYTE_BINARY);
        for (int i = 0; i < grayImage.getWidth(); i++) {
            for (int j = 0; j < grayImage.getHeight(); j++) {
                int pixel = grayImage.getRGB(i, j);
                int intensity = (pixel >> RED_INTENSITY_SHIFT) & 0xff;
                thresholdedImage.setRGB(i, j, intensity > threshold ? 0xFFFFFF : 0x000000);
            }
        }
        return thresholdedImage;
    }

    private BufferedImage toGrayScale(BufferedImage img) {
        System.out.println("  Converting to GrayScale.");
        BufferedImage grayImage = new BufferedImage(
                img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = grayImage.getGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();
        return grayImage;
    }

    private int[] calculateHistogram(BufferedImage image) {
        int[] histogram = new int[HISTOGRAM_SIZE];
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                int pixel = image.getRGB(i, j);
                int intensity = (pixel >> 16) & 0xff;
                histogram[intensity]++;
            }
        }
        return histogram;
    }

    private int findOtsuThreshold(int[] histogram, int totalPixels) {
        float totalPixelIntensity = 0;
        for (int intensity = 0; intensity < HISTOGRAM_SIZE; intensity++) {
            totalPixelIntensity += intensity * histogram[intensity];
        }
        float backgroundPixelIntensity = 0;
        int backgroundWeight = 0;
        int foregroundWeight = 0;
        float maxVariance = 0;
        int optimalThreshold = 0;
        for (int threshold = 0; threshold < HISTOGRAM_SIZE; threshold++) {
            backgroundWeight += histogram[threshold];

            if (backgroundWeight == 0) continue;
            foregroundWeight = totalPixels - backgroundWeight;
            if (foregroundWeight == 0) break;

            backgroundPixelIntensity += (float) (threshold * histogram[threshold]);

            float meanBackgroundIntensity = backgroundPixelIntensity / backgroundWeight;
            float meanForegroundIntensity = (totalPixelIntensity - backgroundPixelIntensity) / foregroundWeight;
            float betweenClassVariance = calculateBetweenClassVariance(backgroundWeight, foregroundWeight, meanBackgroundIntensity, meanForegroundIntensity);

            if (betweenClassVariance > maxVariance) {
                maxVariance = betweenClassVariance;
                optimalThreshold = threshold;
            }
        }
        return optimalThreshold;
    }

    private float calculateBetweenClassVariance(int backgroundWeight, int foregroundWeight, float meanBackgroundIntensity, float meanForegroundIntensity) {
        return (float) ((float) backgroundWeight * (float) foregroundWeight * Math.pow(meanBackgroundIntensity - meanForegroundIntensity, 2));
    }

}








