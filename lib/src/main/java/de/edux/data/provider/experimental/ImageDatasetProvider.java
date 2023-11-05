package de.edux.data.provider.experimental;

import de.edux.data.reader.CSVIDataReader;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import javax.imageio.ImageIO;

public class ImageDatasetProvider {
  private static final File CSV_TRAIN_FILE =
      new File(
          "example"
              + File.separator
              + "datasets"
              + File.separator
              + "fractalitylab-dataset"
              + File.separator
              + "train"
              + File.separator
              + "images.csv");
  private static final File CSV_TEST_FILE =
      new File(
          "example"
              + File.separator
              + "datasets"
              + File.separator
              + "fractalitylab-dataset"
              + File.separator
              + "test"
              + File.separator
              + "images.csv");
  private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
  private final double[][] trainLabels;
  private final double[][] trainFeatures;

  private final double[][] testLabels;
  private final double[][] testFeatures;
  private CSVIDataReader trainCsvDataReader = new CSVIDataReader();

  private CSVIDataReader testCsvDataReader = new CSVIDataReader();

  public ImageDatasetProvider() {
    List<String[]> train_dataRows = trainCsvDataReader.readFile(CSV_TRAIN_FILE, ',');
    List<DataElement> train_dataElements = process(train_dataRows, "train");

    List<String[]> test_dataRows = testCsvDataReader.readFile(CSV_TEST_FILE, ',');
    List<DataElement> test_dataElements = process(test_dataRows, "test");

    // Start concurrent tasks using virtual threads
    Future<double[][]> trainFeaturesFuture = executor.submit(() -> readImages(train_dataElements));
    Future<double[][]> trainLabelsFuture = executor.submit(() -> readLabels(train_dataElements));
    Future<double[][]> testFeaturesFuture = executor.submit(() -> readImages(test_dataElements));
    Future<double[][]> testLabelsFuture = executor.submit(() -> readLabels(test_dataElements));

    try {
      // Retrieve the results of the concurrent tasks
      trainFeatures = trainFeaturesFuture.get();
      trainLabels = trainLabelsFuture.get();
      testFeatures = testFeaturesFuture.get();
      testLabels = testLabelsFuture.get();
    } catch (Exception e) {
      // handle exceptions
      throw new RuntimeException("Error while loading dataset using virtual threads", e);
    } finally {
      executor.shutdown(); // Always remember to shutdown the executor
    }
  }

  public double[][] getTestLabels() {
    return testLabels;
  }

  public double[][] getTestFeatures() {
    return testFeatures;
  }

  private double[][] readLabels(List<DataElement> dataElements) {
    // Sammle alle einzigartigen Labels
    Set<String> uniqueLabels = new HashSet<>();
    for (DataElement element : dataElements) {
      uniqueLabels.add(element.getLabel());
    }

    // Erstelle ein Mapping von Label zu Index
    Map<String, Integer> labelToIndex = new HashMap<>();
    int index = 0;
    for (String label : uniqueLabels) {
      labelToIndex.put(label, index++);
    }

    // Erstelle ein One-Hot-kodiertes Array für Labels
    double[][] labelsOneHot = new double[dataElements.size()][uniqueLabels.size()];
    for (int i = 0; i < dataElements.size(); i++) {
      int labelIndex = labelToIndex.get(dataElements.get(i).getLabel());
      labelsOneHot[i][labelIndex] = 1.0;
    }

    return labelsOneHot;
  }

  private double[][] readImages(List<DataElement> dataElements) {
    // Annahme: Alle Bilder haben die gleiche Größe
    // Die Größe des ersten Bildes wird verwendet, um die Größe des 'flachgelegten' Arrays zu
    // bestimmen
    BufferedImage firstImage;
    try {
      firstImage = ImageIO.read(new File(dataElements.get(0).getImagePath()));
    } catch (IOException e) {
      throw new RuntimeException("Fehler beim Laden des ersten Bildes", e);
    }
    int width = firstImage.getWidth();
    int height = firstImage.getHeight();
    int imageSize = width * height;

    // Array für alle 'flachgelegten' Bilder erstellen
    double[][] flatImages = new double[dataElements.size()][imageSize];

    for (int i = 0; i < dataElements.size(); i++) {
      BufferedImage image;
      try {
        image = ImageIO.read(new File(dataElements.get(i).getImagePath()));
      } catch (IOException e) {
        throw new RuntimeException("Fehler beim Laden von Bild " + i, e);
      }

      // Bildpixel durchlaufen und in eindimensionales Array umwandeln
      double[] flatImage = new double[imageSize];
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int color = image.getRGB(x, y);
          int red = (color >> 16) & 0xff; // Hier nehmen wir nur den Rotkanal
          // Für Graustufenbilder könnte man stattdessen den Durchschnitt der RGB-Werte verwenden
          flatImage[y * width + x] = red / 255.0; // Normalisierung auf den Bereich [0, 1]
        }
      }
      flatImages[i] = flatImage;
    }

    return flatImages;
  }

  private List<DataElement> process(List<String[]> dataRows, String location) {
    List<DataElement> result = new ArrayList<>();
    dataRows.remove(0);
    dataRows.forEach(
        row -> {
          String label = row[1];
          String imagePath =
              "example"
                  + File.separator
                  + "datasets"
                  + File.separator
                  + "fractalitylab-dataset"
                  + File.separator
                  + location
                  + File.separator
                  + "class"
                  + File.separator
                  + label
                  + File.separator
                  + row[0]
                  + ".png";
          DataElement dataElement = new DataElement(label, imagePath);
          result.add(dataElement);
        });

    return result;
  }

  public double[][] getTrainLabels() {
    return trainLabels;
  }

  public double[][] getTrainFeatures() {
    return trainFeatures;
  }

  public CSVIDataReader getTrainCsvDataReader() {
    return trainCsvDataReader;
  }

  public void setTrainCsvDataReader(CSVIDataReader trainCsvDataReader) {
    this.trainCsvDataReader = trainCsvDataReader;
  }
}
