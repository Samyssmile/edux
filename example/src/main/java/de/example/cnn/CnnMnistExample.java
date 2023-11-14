package de.example.cnn;

import de.edux.ml.cnn.SimpleCNN;
import de.edux.ml.cnn.layers.*;
import de.edux.ml.cnn.math.Matrix3D;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class CnnMnistExample {
  public static void main(String[] args) throws IOException {
    String trainImagesPath =
        "E:\\projects\\edux\\example\\datasets\\mnist\\train-images.idx3-ubyte";
    String trainLabelsPath =
        "E:\\projects\\edux\\example\\datasets\\mnist\\train-labels.idx1-ubyte";
    String testImagesPath = "E:\\projects\\edux\\example\\datasets\\mnist\\t10k-images.idx3-ubyte";
    String testLabelsPath = "E:\\projects\\edux\\example\\datasets\\mnist\\t10k-labels.idx1-ubyte";
    // Load MNIST data
    List<Matrix3D> trainImages = loadImages(trainImagesPath, 100);
    List<Matrix3D> trainLabels = loadLabels(trainLabelsPath, 100);

    List<Matrix3D> testImages = loadImages(testImagesPath, 100);
    List<Matrix3D> testLabels = loadLabels(testLabelsPath, 100);

    SimpleCNN model = new SimpleCNN();
    model.train(trainImages, trainLabels, 0.01, 100, testImages, testLabels);
    double accuracy = model.evaluate(testImages, testLabels);
    System.out.println("Accuracy: " + accuracy + "%");
  }

  public static List<Matrix3D> loadImages(String path, int limit) throws IOException {
    try (FileInputStream in = new FileInputStream(path)) {
      byte[] buffer = new byte[4];

      // Reading and skipping the magic number
      in.read(buffer, 0, 4);

      // Reading the number of images
      in.read(buffer, 0, 4);
      int numberOfImages = ByteBuffer.wrap(buffer).getInt();

      // Reading rows and columns
      in.read(buffer, 0, 4);
      int rows = ByteBuffer.wrap(buffer).getInt();
      in.read(buffer, 0, 4);
      int cols = ByteBuffer.wrap(buffer).getInt();

      List<Matrix3D> images = new ArrayList<>();

      for (int i = 0; i < (limit == 0 ? numberOfImages : limit); i++) {
        Matrix3D img =
            new Matrix3D(1, rows, cols); // Assuming Matrix3D is designed to hold image data
        for (int r = 0; r < rows; r++) {
          for (int c = 0; c < cols; c++) {
            // Normalizing each pixel by dividing by 255
            img.set(0, r, c, (in.read() & 0xFF) / 255.0); // Reading each pixel and normalizing
          }
        }
        images.add(img);
      }

      return images;
    }
  }

  public static List<Matrix3D> loadLabels(String path, int limit) throws IOException {
    try (FileInputStream in = new FileInputStream(path)) {
      byte[] buffer = new byte[4];

      // Reading and skipping the magic number
      in.read(buffer, 0, 4);

      // Reading the number of labels
      in.read(buffer, 0, 4);
      int numberOfLabels = ByteBuffer.wrap(buffer).getInt();

      List<Matrix3D> labels = new ArrayList<>();

      for (int i = 0; i < (limit == 0 ? numberOfLabels : limit); i++) {
        Matrix3D label = new Matrix3D(1, 1, 10); // 10 for one-hot encoding of 0-9 digits
        int labelValue = in.read();
        label.set(0, 0, labelValue, 1); // Setting the corresponding index to 1
        labels.add(label);
      }

      return labels;
    }
  }
}
