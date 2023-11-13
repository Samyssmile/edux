package de.edux.ml.cnn;

import de.edux.ml.cnn.layers.*;
import de.edux.ml.cnn.loss.CrossEntropyLossV2;
import de.edux.ml.cnn.math.Matrix3D;
import java.util.List;

public class SimpleCNN {
  private Layer[] layers;

  public SimpleCNN() {
    this.layers =
        new Layer[] {
          new ConvolutionalLayer(8, 3, 1, 1), // 8 Filter, 3x3 Größe, Stride 1, paddind 1
          new ReLuLayer(),
          new MaxPoolingLayer(2, 2), // 2x2 Größe, Stride 2
          new ConvolutionalLayer(16, 3, 1, 1),
          new ReLuLayer(),
          new MaxPoolingLayer(2, 2),
          new FlattenLayer(),
          new DenseLayer(784, 10),
          new SoftmaxLayer()
        };
  }

  public Matrix3D forward(Matrix3D input) {
    Matrix3D output = input;
    for (Layer layer : layers) {
      output = layer.forward(output);
    }
    return output;
  }

  // Vorhersage für ein einzelnes Beispiel
  public Matrix3D predict(Matrix3D input) {
    return forward(input);
  }

  // Bewertung der Netzwerkleistung
  public double evaluate(List<Matrix3D> testImages, List<Matrix3D> testLabels) {
    int correctPredictions = 0;
    for (int i = 0; i < testImages.size(); i++) {
      Matrix3D prediction = predict(testImages.get(i));
      int predictedLabel = argMax(prediction);
      int trueLabel = argMax(testLabels.get(i));

      if (predictedLabel == trueLabel) {
        correctPredictions++;
      }
    }
    return (double) correctPredictions / testImages.size();
  }

  // Hilfsmethode zur Bestimmung des Indexes des größten Werts in der Ausgabe
  private int argMax(Matrix3D matrix) {
    int maxIndex = 0;
    double max = matrix.get(0, 0, 0);
    for (int i = 1; i < matrix.getCols(); i++) {
      if (matrix.get(0, 0, i) > max) {
        max = matrix.get(0, 0, i);
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  // Training des Netzwerks
  public void train(
      List<Matrix3D> trainImages, List<Matrix3D> trainLabels, double learningRate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      double totalLoss = 0.0;

      for (int i = 0; i < trainImages.size(); i++) {
        Matrix3D input = trainImages.get(i);
        Matrix3D trueOutput = trainLabels.get(i);

        // Vorwärtspropagierung
        Matrix3D predictedOutput = forward(input);

        // Berechnen des Verlustes (hier Cross-Entropy)
        CrossEntropyLossV2 lossFunction = new CrossEntropyLossV2();
        totalLoss += lossFunction.computeLoss(predictedOutput, trueOutput);

        // Rückwärtspropagierung
        Matrix3D errorGradient = lossFunction.computeGradient(predictedOutput, trueOutput);
        backward(errorGradient, learningRate);
      }

      System.out.println("Epoch " + (epoch + 1) + ", Loss: " + totalLoss / trainImages.size());
    }
  }

  private void backward(Matrix3D errorGradient, double learningRate) {
    for (int i = layers.length - 1; i >= 0; i--) {
      errorGradient = layers[i].backward(errorGradient, learningRate);
    }
  }
}
