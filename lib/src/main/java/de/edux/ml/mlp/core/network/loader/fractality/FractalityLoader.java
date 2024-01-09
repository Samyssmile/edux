package de.edux.ml.mlp.core.network.loader.fractality;

import de.edux.ml.mlp.core.network.loader.BatchData;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.Stream;
import javax.imageio.ImageIO;

public class FractalityLoader implements Loader {

  private final String imageFolderPath;
  private final String csvLabelDataFile;
  private final int batchLength;
  private final Map<String, String> csvContent;
  private final int imageWidth;
  private final int imageHeight;
  Iterator<Map.Entry<String, String>> csvContentIterator;
  private FractalityMetaData metaData;

  public FractalityLoader(
      String imageFolderPath, String csvLabelDataFile, int batchLength, int width, int height) {
    this.imageWidth = width;
    this.imageHeight = height;
    this.imageFolderPath = imageFolderPath;
    this.csvLabelDataFile = csvLabelDataFile;
    this.batchLength = batchLength;

    csvContent = getCsvContent(csvLabelDataFile);
    csvContentIterator = csvContent.entrySet().iterator();
  }

  private Map<String, String> getCsvContent(String csvLabelDataFile) {
    Map<String, String> csvContent = new HashMap<>();

    try (Stream<String> stream = Files.lines(Paths.get(csvLabelDataFile))) {
      stream
          .skip(1)
          .forEach(
              line -> {
                String[] parts = line.split(",");
                if (parts.length >= 2) {
                  csvContent.put(parts[0], parts[1]);
                }
              });
    } catch (IOException e) {
      e.printStackTrace();
    }

    return csvContent;
  }

  @Override
  public MetaData open() {
    return readMetaData();
  }

  private MetaData readMetaData() {
    this.metaData = new FractalityMetaData();
    metaData.setNumberItems(csvContent.size());
    metaData.setInputSize(imageWidth * imageHeight);
    metaData.setNumberOfClasses(6);
    metaData.setNumberBatches((int) Math.ceil(metaData.getNumberItems() / batchLength));
    metaData.setBatchLength(batchLength);

    return metaData;
  }

  @Override
  public void close() {
    this.metaData = null;
  }

  @Override
  public MetaData getMetaData() {
    return metaData;
  }

  @Override
  public BatchData readBatch() {
    BatchData batchData = new FractalityBatchData();

    int inputsRead = readInputBatch(batchData);
    metaData.setItemsRead(inputsRead);

    return batchData;
  }

  @Override
  public void reset() {
    csvContentIterator = csvContent.entrySet().iterator();
  }

  private int readInputBatch(BatchData batchData) {
    var numberToRead =
        Math.min(
            metaData.getBatchLength(), (metaData.getInputSize() - metaData.getTotalItemsRead()));

    double[] dataInputs = new double[metaData.getInputSize() * metaData.getBatchLength()];
    double[] dataExpected = new double[metaData.getNumberOfClasses() * metaData.getBatchLength()];
    for (int i = 0; i < numberToRead; i++) {
      if (csvContentIterator.hasNext()) {
        Map.Entry<String, String> entry = csvContentIterator.next();
        String imagePath =
            imageFolderPath
                + File.separator
                + entry.getValue()
                + File.separator
                + entry.getKey()
                + ".png";

        try {
          BufferedImage image = ImageIO.read(new File(imagePath));

          int indexInputs = 0;

          for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
              Color color = new Color(image.getRGB(x, y));
              dataInputs[indexInputs++] = colorToDouble(color);
            }
          }
          batchData.setInputBatch(dataInputs);

          int indexExpected = 0;
          dataExpected[indexExpected++] = fractalityToDouble(entry.getValue());
          batchData.setExpectedBatch(dataExpected);
        } catch (IOException e) {
          e.printStackTrace();
          return 0;
        }
      }
    }
    return dataInputs.length / metaData.getInputSize();
  }

  private double fractalityToDouble(String value) {
    switch (value) {
      case "mandelbrot":
        return 1d;
      case "sierpinski_gasket":
        return 2d;
      case "julia":
        return 3d;
      case "burningship":
        return 4d;
      case "tricorn":
        return 5d;
      case "newton":
        return 6d;
      default:
        return -1;
    }
  }

  private double colorToDouble(Color color) {
    return (color.getRed() + color.getGreen() + color.getBlue()) / 3.0 / 255.0;
  }

  public Map<String, String> getCsvContent() {
    return csvContent;
  }
}
