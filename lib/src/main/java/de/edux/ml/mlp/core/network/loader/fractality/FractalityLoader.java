package de.edux.ml.mlp.core.network.loader.fractality;

import de.edux.ml.mlp.core.network.loader.AbstractBatchData;
import de.edux.ml.mlp.core.network.loader.AbstractMetaData;
import de.edux.ml.mlp.core.network.loader.BatchData;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.exceptions.LoaderException;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class FractalityLoader implements Loader {
  private final int inputHeight;
  private final int inputWeight;
  private String imageDirectory;
  private String labelFileName;
  private int batchSize;
  private List<String> imagePaths; // Liste der Bildpfade
  private List<Integer> labels;    // Liste der zugehörigen Labels
  private int currentIndex;        // Aktueller Index für die Batch-Verarbeitung
  private Map<String, Integer> classToIndex; // Mapping von Klassenname zu Label-Index
  private Map<String, String> imageIdToPath; // Mapping von Image ID zu Bildpfad
  private FractalityMetaData metaData;
  private Lock readLock = new ReentrantLock();

  public FractalityLoader(String imageDirectory, String labelFileName, int batchSize, int inputHeight, int inputWeight) {
    this.imageDirectory = imageDirectory;
    this.labelFileName = labelFileName;
    this.batchSize = batchSize;
    this.classToIndex = new HashMap<>();
    this.inputHeight = inputHeight;
    this.inputWeight = inputWeight;
  }

  @Override
  public MetaData open() {
    return readMetaData();
  }

  @Override
  public void close() {
    imagePaths = null;
    labels = null;
    metaData = null;
    currentIndex = 0;
  }

  @Override
  public MetaData getMetaData() {
    return metaData;
  }

  @Override
  public BatchData readBatch() {
    readLock.lock();
    try {
      if (currentIndex >= imagePaths.size()) {
        return null; // Keine weiteren Daten
      }
      int endIndex = Math.min(currentIndex + batchSize, imagePaths.size());
      List<String> batchImagePaths = imagePaths.subList(currentIndex, endIndex);
      List<Integer> batchLabels = labels.subList(currentIndex, endIndex);
      currentIndex = endIndex;

      FractalityBatchData batchData = new FractalityBatchData();
      int itemsRead = readInputBatch(batchData, batchImagePaths, batchLabels);
      metaData.setItemsRead(itemsRead);
      return batchData;
    } finally {
      readLock.unlock();
    }
  }

  private MetaData readMetaData() {
    metaData = new FractalityMetaData();
    imagePaths = new ArrayList<>();
    labels = new ArrayList<>();
    classToIndex = new HashMap<>();

    // Bildpfade einlesen und Image ID zu Pfad mappen
    buildImageIdToPathMap();

    try (BufferedReader br = new BufferedReader(new FileReader(labelFileName))) {
      String line;
      // Header überspringen
      br.readLine();
      while ((line = br.readLine()) != null) {
        String[] tokens = line.split(",");
        if (tokens.length != 2) {
          throw new LoaderException("Ungültige Zeile in CSV-Datei: " + line);
        }
        String imageId = tokens[0];
        String labelName = tokens[1];

        // Klassenname zu Index mappen
        classToIndex.computeIfAbsent(labelName, k->(labelToIndex(k)));
        int labelIndex = classToIndex.get(labelName);

        // Bildpfad aus Mapping abrufen
        String imagePath = imageIdToPath.get(imageId);
        if (imagePath == null) {
          throw new LoaderException("Bilddatei nicht gefunden für Image ID: " + imageId);
        }

        imagePaths.add(imagePath);
        labels.add(labelIndex);
      }

      metaData.setNumberItems(imagePaths.size());
      metaData.setNumberOfClasses(classToIndex.size());
      metaData.setInputSize(this.inputHeight * this.inputWeight);
      metaData.setNumberBatches((int) Math.ceil((double) imagePaths.size() / batchSize));
      metaData.setClassToIndex(classToIndex);
    } catch (IOException e) {
      throw new LoaderException("Fehler beim Lesen der CSV-Datei: " + labelFileName);
    }
    return metaData;
  }

  private int labelToIndex(String label) {

    if (label.equalsIgnoreCase("tricorn")){
        return 0;
    }
    if (label.equalsIgnoreCase("burningship")){
        return 1;
    }
    if (label.equalsIgnoreCase("mandelbrot")){
        return 2;
    }
    if (label.equalsIgnoreCase("julia")){
        return 3;
    }
    if (label.equalsIgnoreCase("sierpinski")){
        return 4;
    }
    if (label.equalsIgnoreCase("newton")){
      return 5;
    }

    throw new LoaderException("Ungültiges Label: " + label);
  }

  private void buildImageIdToPathMap() {
    imageIdToPath = new HashMap<>();
    File classDir = new File(imageDirectory + File.separator + "class");
    if (classDir.exists() && classDir.isDirectory()) {
      File[] classDirs = classDir.listFiles();
      if (classDirs != null) {
        for (File dir : classDirs) {
          if (dir.isDirectory()) {
            File[] imageFiles = dir.listFiles((d, name) -> name.endsWith(".png"));
            if (imageFiles != null) {
              for (File imageFile : imageFiles) {
                String filename = imageFile.getName();
                if (filename.endsWith(".png")) {
                  String imageId = filename.substring(0, filename.length() - 4); // ".png" entfernen
                  imageIdToPath.put(imageId, imageFile.getAbsolutePath());
                }
              }
            }
          }
        }
      }
    } else {
      throw new LoaderException("Klassenverzeichnis nicht gefunden: " + classDir.getAbsolutePath());
    }
  }

  private int readInputBatch(FractalityBatchData batchData, List<String> batchImagePaths, List<Integer> batchLabels) {
    int numberToRead = batchImagePaths.size();
    int inputSize = metaData.getInputSize();
    int expectedSize = metaData.getNumberOfClasses();

    double[] inputData = new double[numberToRead * inputSize];
    double[] expectedData = new double[numberToRead * expectedSize];

    for (int i = 0; i < numberToRead; i++) {
      String imagePath = batchImagePaths.get(i);
      int labelIndex = batchLabels.get(i);

      // Bild einlesen und in Double-Array konvertieren
      double[] imageData = readImage(imagePath, this.inputWeight, this.inputHeight);
      System.arraycopy(imageData, 0, inputData, i * inputSize, inputSize);

      // One-Hot-Encoding für Label
      expectedData[i * expectedSize + labelIndex] = 1.0;
    }
    batchData.setInputBatch(inputData);
    batchData.setExpectedBatch(expectedData);
    return numberToRead;
  }

  private double[] readImage(String imagePath, int width, int height) {
    double[] imageData = new double[width * height];
    try {
      BufferedImage img = ImageIO.read(new File(imagePath));
      if (img.getWidth() != width || img.getHeight() != height) {
        // Bild skalieren
        Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        img = resized;
      }
      // Bilddaten auslesen
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int rgb = img.getRGB(x, y);
          int gray = (rgb >> 16) & 0xFF; // Graustufenwert
          imageData[y * width + x] = gray / 256.0;
        }
      }
    } catch (IOException e) {
      throw new LoaderException("Fehler beim Lesen der Bilddatei: " + imagePath);
    }
    return imageData;
  }

  // Innere Klasse für MetaData
  public class FractalityMetaData extends AbstractMetaData {
    private Map<String, Integer> classToIndex;

    public Map<String, Integer> getClassToIndex() {
      return classToIndex;
    }

    public void setClassToIndex(Map<String, Integer> classToIndex) {
      this.classToIndex = classToIndex;
    }

    @Override
    public void setItemsRead(int itemsRead) {
      super.setItemsRead(itemsRead);
      super.setTotalItemsRead(super.getTotalItemsRead() + itemsRead);
    }
  }

  // Innere Klasse für BatchData
  public class FractalityBatchData extends AbstractBatchData {
    // Kann erweitert werden, falls nötig
  }
}
