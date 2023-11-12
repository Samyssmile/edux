package de.example.cnn;

import de.edux.ml.cnn.CNNModel;
import de.edux.ml.cnn.layers.*;
import de.edux.ml.cnn.math.Matrix;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
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
    List<Matrix> trainImages = loadImages(trainImagesPath);
    List<Matrix> trainLabels = loadLabels(trainLabelsPath);

    List<Matrix> testImages = loadImages(testImagesPath);
    List<Matrix> testLabels = loadLabels(testLabelsPath);

    // Initialize the CNN model
    CNNModel model = new CNNModel(0.1);
    model.addLayer(new ConvolutionalLayer(24, 3));
    model.addLayer(new ReLULayer());
    model.addLayer(new MaxPoolingLayer(2, 2));
    model.addLayer(new ConvolutionalLayer(48, 3));
    model.addLayer(new ReLULayer());
    model.addLayer(new MaxPoolingLayer(2, 2));
    model.addLayer(new FlattenLayer());
    model.addLayer(new FullyConnectedLayer(25, 10));
    model.addLayer(new SoftmaxLayer());

    // Train the model
    model.train(trainImages, trainLabels, testImages, testLabels, 20); // 5 epochs
    model.evaluate(testImages, testLabels);
  }

  private static List<Matrix> loadImages(String path) {
    try (FileInputStream fis = new FileInputStream(path);
        FileChannel channel = fis.getChannel()) {

      // Lesen des Headers
      ByteBuffer buffer = ByteBuffer.allocate(16);
      channel.read(buffer);
      buffer.flip();

      // Ignorieren der Magic Number
      buffer.getInt();

      int numImages = buffer.getInt();
      int numRows = buffer.getInt();
      int numCols = buffer.getInt();

      // Vorbereiten der Matrix-Array
      Matrix[] images = new Matrix[numImages];

      for (int i = 0; i < numImages; i++) {
        buffer = ByteBuffer.allocate(numRows * numCols);
        channel.read(buffer);
        buffer.flip();

        double[][] data = new double[numRows][numCols];
        for (int r = 0; r < numRows; r++) {
          for (int c = 0; c < numCols; c++) {
            data[r][c] = (double) (buffer.get() & 0xFF) / 255.0; // Normalisierung der Pixelwerte
          }
        }
        images[i] = new Matrix(data, numRows, numCols);
      }
      return List.of(images);
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static List<Matrix> loadLabels(String path) {
    try (FileInputStream fis = new FileInputStream(path);
        FileChannel channel = fis.getChannel()) {

      // Lesen des Headers
      ByteBuffer buffer = ByteBuffer.allocate(8);
      channel.read(buffer);
      buffer.flip();

      // Ignorieren der Magic Number
      buffer.getInt();

      int numLabels = buffer.getInt();

      // Vorbereiten der Matrix-Array
      Matrix[] labels = new Matrix[numLabels];

      for (int i = 0; i < numLabels; i++) {
        buffer = ByteBuffer.allocate(1);
        channel.read(buffer);
        buffer.flip();

        int label = (int) buffer.get();
        double[][] labelData = new double[10][1]; // 10 classes for digits 0-9
        labelData[label][0] = 1.0; // One-hot encoding
        labels[i] = new Matrix(labelData, 10, 1);
      }
      return List.of(labels);
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static Matrix[] loadImagesAsMatrix(String path) {
    try (FileInputStream fis = new FileInputStream(path);
        FileChannel channel = fis.getChannel()) {

      // Lesen des Headers
      ByteBuffer buffer = ByteBuffer.allocate(16);
      channel.read(buffer);
      buffer.flip();

      // Ignorieren der Magic Number
      buffer.getInt();

      int numImages = buffer.getInt();
      int numRows = buffer.getInt();
      int numCols = buffer.getInt();

      // Vorbereiten der Matrix-Array
      Matrix[] images = new Matrix[numImages];

      for (int i = 0; i < numImages; i++) {
        buffer = ByteBuffer.allocate(numRows * numCols);
        channel.read(buffer);
        buffer.flip();

        double[][] data = new double[numRows][numCols];
        for (int r = 0; r < numRows; r++) {
          for (int c = 0; c < numCols; c++) {
            data[r][c] = (double) (buffer.get() & 0xFF) / 255.0; // Normalisierung der Pixelwerte
          }
        }
        images[i] = new Matrix(data, numRows, numCols);
      }
      return images;
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }

  private static Matrix[] loadLabelsAsMAtrix(String path) {
    try (FileInputStream fis = new FileInputStream(path);
        FileChannel channel = fis.getChannel()) {

      // Lesen des Headers
      ByteBuffer buffer = ByteBuffer.allocate(8);
      channel.read(buffer);
      buffer.flip();

      // Ignorieren der Magic Number
      buffer.getInt();

      int numLabels = buffer.getInt();

      // Vorbereiten der Matrix-Array
      Matrix[] labels = new Matrix[numLabels];

      for (int i = 0; i < numLabels; i++) {
        buffer = ByteBuffer.allocate(1);
        channel.read(buffer);
        buffer.flip();

        int label = (int) buffer.get();
        double[][] labelData = new double[10][1]; // 10 classes for digits 0-9
        labelData[label][0] = 1.0; // One-hot encoding
        labels[i] = new Matrix(labelData, 10, 1);
      }
      return labels;
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }
}
