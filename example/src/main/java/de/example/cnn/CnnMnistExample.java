package de.example.cnn;

import de.edux.ml.cnn.Network;
import de.edux.ml.cnn.NetworkBuilder;
import de.edux.ml.cnn.core.Tensor;
import de.edux.ml.cnn.layers.HiddenLayer;
import de.edux.ml.cnn.layers.OutputLayer;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class CnnMnistExample {
  private static BufferedImage sampleImage;

  private static final int IMAGE_SIZE = 28 * 28; // MNIST images are 28x28


  public static void main(String[] args) throws IOException {
    String trainImagesPath = "mnist\\train-images.idx3-ubyte";
    String trainLabelsPath = "mnist\\train-labels.idx1-ubyte";
    String testImagesPath = "mnist\\t10k-images.idx3-ubyte";
    String testLabelsPath = "mnist\\t10k-labels.idx1-ubyte";

    Tensor trainImages = loadMnistImages(trainImagesPath, 2000);
    Tensor trainLabels = loadMnistLabels(trainLabelsPath, 1000);

    NetworkBuilder builder = new NetworkBuilder();
    builder.addLayer(new HiddenLayer(IMAGE_SIZE, 100));
    builder.addLayer(new HiddenLayer(100, 10));
    builder.addLayer(new OutputLayer(10, 10));

    Network network = builder.build();
    network.train(trainImages, trainLabels, 10, 100, 0.1);
  }

  private static Tensor loadMnistImages(String path, int limit) throws IOException {
    try (DataInputStream dis = new DataInputStream(new FileInputStream(path))) {
      int magicNumber = dis.readInt();
      int numberOfImages = Math.min(dis.readInt(), limit);
      int rows = dis.readInt();
      int cols = dis.readInt();

      Tensor images = new Tensor(numberOfImages, IMAGE_SIZE);
      for (int i = 0; i < numberOfImages; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
          // Normalisiere die Pixelwerte auf den Bereich [0,1]
          images.set(i, j, dis.readUnsignedByte() / 255.0);
        }
      }

      return images;
    }
  }

  private static Tensor loadMnistLabels(String path, int limit) throws IOException {
    try (DataInputStream dis = new DataInputStream(new FileInputStream(path))) {
      int magicNumber = dis.readInt();
      int numberOfLabels = Math.min(dis.readInt(), limit);

      Tensor labels = new Tensor(numberOfLabels, 10); // 10 for one-hot encoding of labels 0-9
      for (int i = 0; i < numberOfLabels; i++) {
        int label = dis.readUnsignedByte();
        labels.set(i, label, 1.0);
      }

      return labels;
    }
  }

}
