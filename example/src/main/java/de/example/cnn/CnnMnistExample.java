package de.example.cnn;

import de.edux.ml.cnn.SimpleCNN;
import de.edux.ml.cnn.layers.*;
import de.edux.ml.cnn.math.Matrix3D;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class CnnMnistExample {
  private static BufferedImage sampleImage;

  public static void main(String[] args) throws IOException {
    String trainImagesPath =
        "mnist\\train-images.idx3-ubyte";
    String trainLabelsPath =
        "mnist\\train-labels.idx1-ubyte";
    String testImagesPath = "mnist\\t10k-images.idx3-ubyte";
    String testLabelsPath = "mnist\\t10k-labels.idx1-ubyte";
    // Load MNIST data
    List<Matrix3D> trainImages = loadImages(trainImagesPath, 1000);
    List<Matrix3D> trainLabels = loadLabels(trainLabelsPath, 1000);

    List<Matrix3D> testImages = loadImages(testImagesPath, 1000);
    List<Matrix3D> testLabels = loadLabels(testLabelsPath, 1000);

    SimpleCNN model = new SimpleCNN();
    model.train(trainImages, trainLabels, 0.01, 20, testImages, testLabels);
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
        Matrix3D img = new Matrix3D(1, rows, cols);
        BufferedImage bufferedImage = new BufferedImage(cols, rows, BufferedImage.TYPE_BYTE_GRAY);

        for (int r = 0; r < rows; r++) {
          for (int c = 0; c < cols; c++) {
            int pixel = in.read() & 0xFF;
            img.set(0, r, c, pixel / 255.0);
            int gray = 255 - pixel;
            bufferedImage.setRGB(c, r, (gray << 16) | (gray << 8) | gray);
          }
        }

        images.add(img);

        // Display the first three images
/*        if (i < 5) {
          displayImage(bufferedImage, "Image " + (i + 1));
          CnnMnistExample.sampleImage = bufferedImage;
        }*/
      }

      return images;
    }
  }

  private static void displayImage(BufferedImage bufferedImage, String s) {
    /*If Desktop show image*/
    if (Desktop.isDesktopSupported()) {
      try {
        File outputfile = new File(s + ".png");
        ImageIO.write(bufferedImage, "png", outputfile);
        Desktop.getDesktop().open(outputfile);
      } catch (IOException e) {
        e.printStackTrace();
      }
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
  /*      if (i < 5) {
          System.out.println(labelValue);
        }*/
        label.set(0, 0, labelValue, 1); // Setting the corresponding index to 1
        labels.add(label);
      }

      return labels;
    }
  }
}
