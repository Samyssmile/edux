package de.example.cnn;

import de.edux.ml.cnn.SimpleCNN;
import de.edux.ml.cnn.layers.*;
import de.edux.ml.cnn.math.Matrix3D;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CnnMnistExample {
  public static void main(String[] args) {
    String trainImagesPath =
        "E:\\projects\\edux\\example\\datasets\\mnist\\train-images.idx3-ubyte";
    String trainLabelsPath =
        "E:\\projects\\edux\\example\\datasets\\mnist\\train-labels.idx1-ubyte";
    String testImagesPath = "E:\\projects\\edux\\example\\datasets\\mnist\\t10k-images.idx3-ubyte";
    String testLabelsPath = "E:\\projects\\edux\\example\\datasets\\mnist\\t10k-labels.idx1-ubyte";

    // Load MNIST data
    List<Matrix3D> trainImages = loadImages(trainImagesPath);
    List<Matrix3D> trainLabels = loadLabels(trainLabelsPath);

    List<Matrix3D> testImages = loadImages(testImagesPath);
    List<Matrix3D> testLabels = loadLabels(testLabelsPath);

    SimpleCNN model = new SimpleCNN();
    model.train(trainImages, trainLabels, 0.0001, 100, testImages, testLabels);
    double accuracy = model.evaluate(testImages, testLabels);
    System.out.println("Accuracy: " + accuracy + "%");
  }

  private static List<Matrix3D> loadImages(String path) {
    try (DataInputStream dis =
        new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
      int magic = dis.readInt();
      int numImages = dis.readInt();
      int numRows = dis.readInt();
      int numCols = dis.readInt();

      List<Matrix3D> images = new ArrayList<>();

      for (int i = 0; i < numImages; i++) {
        Matrix3D image = new Matrix3D(1, numRows, numCols); // Tiefe 1 für einfarbige Bilder
        for (int r = 0; r < numRows; r++) {
          for (int c = 0; c < numCols; c++) {
            image.set(0, r, c, (dis.readUnsignedByte() / 255.0)); // Normalisierung der Pixelwerte
          }
        }
        images.add(image);
      }
      return images;
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static List<Matrix3D> loadLabels(String path) {
    try (DataInputStream dis =
        new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
      int magic = dis.readInt();
      int numLabels = dis.readInt();

      List<Matrix3D> labels = new ArrayList<>();

      for (int i = 0; i < numLabels; i++) {
        Matrix3D label = new Matrix3D(1, 1, 10); // 10 Klassen, 1x1x10 Matrix

        // Manuell alle Werte auf 0 setzen
        for (int c = 0; c < 10; c++) {
          label.set(0, 0, c, 0.0);
        }

        int lbl = dis.readUnsignedByte();
        label.set(0, 0, lbl, 1.0); // Setzen des entsprechenden Indexes auf 1 für One-Hot-Encoding
        labels.add(label);
      }
      return labels;
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }
}
