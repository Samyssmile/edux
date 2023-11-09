package de.edux;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class ImageResizer {

    public static void resizeImagesInDirectory(String directoryPath, int width, int height) {
        try (Stream<Path> paths = Files.walk(Paths.get(directoryPath))) {
            paths.parallel() // Verwendet einen parallelen Stream
                    .filter(Files::isRegularFile) // Filtert nur reguläre Dateien (keine Verzeichnisse)
                    .forEach(path -> resizeAndSaveImage(path, width, height));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void resizeAndSaveImage(Path imagePath, int width, int height) {
        try {
            // Das ursprüngliche Bild einlesen
            BufferedImage originalImage = ImageIO.read(imagePath.toFile());

            // Ein neues, verkleinertes Bild erstellen
            BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            Graphics2D graphics2D = resizedImage.createGraphics();

            // Qualitätsparameter setzen
            graphics2D.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            graphics2D.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_QUALITY);
            graphics2D.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
            graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Das Bild zeichnen
            graphics2D.drawImage(originalImage, 0, 0, width, height, null);
            graphics2D.dispose();

            // Das verkleinerte Bild speichern
            String fileName = imagePath.getFileName().toString();
            File outputdir = new File("resized");
            if (!outputdir.exists()) outputdir.mkdir();
            File outputfile = new File(outputdir, fileName);
            ImageIO.write(resizedImage, "png", outputfile);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
    String directoryPath =
        "C:\\Users\\windo\\Pictures\\dataset\\class\\tricorn"; // Der Pfad zum Verzeichnis mit den
                                                               // Bildern
        int newWidth = 100; // Die neue Breite
        int newHeight = 100; // Die neue Höhe
        resizeImagesInDirectory(directoryPath, newWidth, newHeight);
    }
}
