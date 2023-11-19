package de.example.augmentation;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import de.edux.augmentation.effects.ColorEqualizationAugmentation;
import de.edux.augmentation.effects.ResizeAugmentation;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class SingleImageAugmentationExample {
    private static final String IMAGE_PATH = "images/cyborg.png";
    private static final Integer TARGET_WIDTH = 250;
    private static final Integer TARGET_HEIGHT = 250;

    public static void main(String[] args) throws IOException {
        // Get Buffered Image from image file

        BufferedImage bufferedImage = loadTestImage(IMAGE_PATH);


        AugmentationSequence augmentationSequence=
                new AugmentationBuilder()
                        .addAugmentation(new ResizeAugmentation(TARGET_WIDTH,TARGET_HEIGHT))
                        .addAugmentation(new ColorEqualizationAugmentation())
                        .build();

        BufferedImage augmentedImage=augmentationSequence.applyTo(bufferedImage);

        System.out.println(augmentedImage);


    }

    public static BufferedImage loadTestImage(String path) throws IOException {
        var resourcePath = path;
        var imageStream =
                SingleImageAugmentationExample.class.getClassLoader().getResourceAsStream(resourcePath);
        if (imageStream == null) {
            throw new IOException("Cannot find resource: " + resourcePath);
        }
        return ImageIO.read(imageStream);
    }
}
