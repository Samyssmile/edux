package de.edux.augmentation.effects;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import static de.edux.augmentation.AugmentationTestUtils.loadTestImage;
import static de.edux.augmentation.AugmentationTestUtils.openImageInPreview;

public class BrightnessAugmentationTest {

    private BufferedImage originalImage;
    private BufferedImage augmentedImage;


    @AfterEach
    public void checkConformity() throws IOException {
        int[] originalPixels =
                originalImage.getRGB(0, 0, originalImage.getWidth(), originalImage.getHeight(), null, 0, originalImage.getWidth());
        int[] augmentedPixels =
                augmentedImage.getRGB(
                        0,
                        0,
                        augmentedImage.getWidth(),
                        augmentedImage.getHeight(),
                        null,
                        0,
                        augmentedImage.getWidth());

        assertNotNull(augmentedImage, "Augmented image should not be null.");

        assertEquals(
                originalImage.getWidth(),
                augmentedImage.getWidth(),
                "Augmented image width should match the specified width.");
        assertEquals(
                originalImage.getHeight(),
                augmentedImage.getHeight(),
                "Augmented image height should match the specified height.");
        assertFalse(
                Arrays.equals(originalPixels, augmentedPixels),
                "The augmented image should differ from the original.");

        Path outputPath = Paths.get("augmented.png");
        ImageIO.write(augmentedImage, "png", outputPath.toFile());

        assertTrue(Files.exists(outputPath), "Output image file should exist.");
        assertTrue(Files.size(outputPath) > 0, "Output image file should not be empty.");
    }
    @Test
    public void shouldIncreaseBrightness() throws IOException, InterruptedException {
        originalImage = loadTestImage("augmentation" + File.separator + "fireworks.png");

        double testValue = 0.5;
        BrightnessAugmentation augmentation = new BrightnessAugmentation(testValue);
        augmentedImage = augmentation.apply(originalImage);

        for (int y = 0; y < originalImage.getHeight(); y++){
            for (int x = 0; x < originalImage.getWidth(); x++){
                assertTrue(originalImage.getRGB(x,y) <= augmentedImage.getRGB(x,y));
            }
        }

        openImageInPreview(originalImage);
        openImageInPreview(augmentedImage);
    }

    @Test
    public void shouldDecreaseBrightness() throws IOException, InterruptedException {
        originalImage = loadTestImage("augmentation" + File.separator + "national-park.png");

        double testValue= -0.5;
        BrightnessAugmentation augmentation = new BrightnessAugmentation(testValue);
        augmentedImage = augmentation.apply(originalImage);

        for (int y = 0; y < originalImage.getHeight(); y++){
            for (int x = 0; x < originalImage.getWidth(); x++){
                assertTrue(originalImage.getRGB(x,y) >= augmentedImage.getRGB(x,y));
            }
        }

        openImageInPreview(originalImage);
        openImageInPreview(augmentedImage);
    }
}
