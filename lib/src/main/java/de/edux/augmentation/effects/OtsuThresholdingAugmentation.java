package de.edux.augmentation.effects;

import de.edux.augmentation.core.AbstractAugmentation;

import java.awt.*;
import java.awt.image.BufferedImage;

public class OtsuThresholdingAugmentation extends AbstractAugmentation {


    @Override
    public BufferedImage apply(BufferedImage image) {
        BufferedImage grayImage = toGrayScale((image));
        int[] histogram = calculateHistogram(grayImage);
        int threshold = findOtsuThreshold(histogram, grayImage.getWidth() * grayImage.getHeight());
        BufferedImage thresholdedImage = new BufferedImage(grayImage.getWidth(), grayImage.getHeight(), BufferedImage.TYPE_BYTE_BINARY);
        for (int i = 0; i < grayImage.getWidth(); i++) {
            for (int j = 0; j < grayImage.getHeight(); j++) {
                int pixel = grayImage.getRGB(i, j);
                int intensity = (pixel >> 16) & 0xff;
                thresholdedImage.setRGB(i, j, intensity > threshold ? 0xFFFFFF : 0x000000);
            }
        }
        return thresholdedImage;
    }


    /**
     * Convert it first to Grayscale*
     */
    public BufferedImage toGrayScale(BufferedImage img) {
        System.out.println("  Converting to GrayScale.");
        BufferedImage grayImage = new BufferedImage(
                img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = grayImage.getGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();
        return grayImage;
    }

    private static int[] calculateHistogram(BufferedImage image) {
        int[] histogram = new int[256];
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                int pixel = image.getRGB(i, j);
                int intensity = (pixel >> 16) & 0xff;
                histogram[intensity]++;
            }
        }
        return histogram;
    }

    private static int findOtsuThreshold(int[] histogram, int totalPixels) {
        float totalPixelIntensity = 0;
        for (int intensity = 0; intensity < 256; intensity++) {
            totalPixelIntensity += intensity * histogram[intensity];
        }
        float backgroundPixelIntensity = 0;
        int backgroundWeight = 0;
        int foregroundWeight = 0;
        float maxVariance = 0;
        int optimalThreshold = 0;
        for (int threshold = 0; threshold < 256; threshold++) {
            backgroundWeight += histogram[threshold];
            if (backgroundWeight == 0) continue;
            foregroundWeight = totalPixels - backgroundWeight;
            if (foregroundWeight == 0) break;
            backgroundPixelIntensity += (float) (threshold * histogram[threshold]);

            float meanBackgroundIntensity = backgroundPixelIntensity / backgroundWeight;
            float meanForegroundIntensity = (totalPixelIntensity - backgroundPixelIntensity) / foregroundWeight;
            // Calculate the between-class variance
            float betweenClassVariance = (float) ((float) backgroundWeight
                    * (float) foregroundWeight
                    * Math.pow(meanBackgroundIntensity - meanForegroundIntensity, 2));
            // Update the maximum variance and optimal
            // threshold if the current between-class variance is greater
            if (betweenClassVariance > maxVariance) {
                maxVariance = betweenClassVariance;
                optimalThreshold = threshold;
            }
        }
        return optimalThreshold;
    }
}
