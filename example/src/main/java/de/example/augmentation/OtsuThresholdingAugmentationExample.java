package de.example.augmentation;

import de.edux.augmentation.effects.OtsuThresholdingAugmentation;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class OtsuThresholdingAugmentationExample {

    private static final String IMAGE_PATH = "images" + File.separator + "cyborg.png";

    public static void main(String[] args) throws IOException {
        BufferedImage bufferedImage = loadTestImage(IMAGE_PATH);
        OtsuThresholdingAugmentation otsuThreshold = new OtsuThresholdingAugmentation();
        BufferedImage otsoImage = otsuThreshold.apply(bufferedImage);
        display(bufferedImage, "Original Image");
        display(otsoImage, "After Otsu Thresholing");
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

    public static void display(BufferedImage img, String title) {
        JFrame frame = new JFrame(title);
        JLabel label = new JLabel();
        frame.setSize(img.getWidth(), img.getHeight());
        label.setIcon(new ImageIcon(img));
        frame.getContentPane().add(label, BorderLayout.CENTER);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }

}
