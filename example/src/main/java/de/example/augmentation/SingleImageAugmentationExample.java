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
    private static final File IMAGE_FILE =
            new File(
                    "example"
                            + File.separator
                            + "datasets"
                            + File.separator
                            + "images"
                            + File.separator
                            + "image.jpeg");
    private static final Integer TARGET_WIDTH = 250;
    private static final Integer TARGET_HEIGHT = 250;

    public static void main(String[] args) throws IOException {
        // Get Buffered Image from image file
        System.out.println("<---- Starting Augmentation ----> ");

        BufferedImage bufferedImage = ImageIO.read(IMAGE_FILE);
        System.out.println("<---- Image Before Augmentation ----> ");
        System.out.println(bufferedImage);

        AugmentationSequence augmentationSequence=
                new AugmentationBuilder()
                        .addAugmentation(new ResizeAugmentation(TARGET_WIDTH,TARGET_HEIGHT))
                        .addAugmentation(new ColorEqualizationAugmentation())
                        .build();

        BufferedImage augmentedImage=augmentationSequence.applyTo(bufferedImage);
        System.out.println("<---- Image After Augmentation ----> ");
        System.out.println(augmentedImage);
        System.out.println("<---- Ending Augmentation ----> ");

    }
}
