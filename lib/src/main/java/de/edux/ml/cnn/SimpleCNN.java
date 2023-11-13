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
          new ConvolutionalLayer(8, 3, 1, 1, 1), // inputDepth = 1 GrayScale Image
          new ReLuLayer(),
          new MaxPoolingLayer(2, 2),
          /*          new ConvolutionalLayer(
              16, 3, 1, 1, 8), // inputDepth = 8 from previous layer (numFilter = 8 was outputDepth)
          new ReLuLayer(),
          new MaxPoolingLayer(2, 2),*/
          new FlattenLayer(),
          new DenseLayer(784 * 2, 10),
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

  public void train(
      List<Matrix3D> trainImages,
      List<Matrix3D> trainLabels,
      double learningRate,
      int epochs,
      List<Matrix3D> testImages,
      List<Matrix3D> testLabels) {

    CrossEntropyLossV2 lossFunction = new CrossEntropyLossV2();

    for (int epoch = 0; epoch < epochs; epoch++) {
      double totalLoss = 0.0;

      // Durchgehen aller Trainingsdaten
      for (int i = 0; i < trainImages.size(); i++) {
        Matrix3D input = trainImages.get(i);
        Matrix3D trueOutput = trainLabels.get(i);

        // Forward Pass
        Matrix3D predictedOutput = forward(input);

        // Berechnung des Verlustes
        double loss = lossFunction.computeLoss(predictedOutput, trueOutput);
        totalLoss += loss;

        // Berechnung des Gradienten des Verlustes
        Matrix3D errorGradient = lossFunction.computeGradient(predictedOutput, trueOutput);

        // Backward Pass
        backward(errorGradient, learningRate);
      }

      // Durchschnittlichen Verlust für diese Epoche berechnen
      double averageLoss = totalLoss / trainImages.size();

      // Evaluierung des Netzwerks auf den Testdaten
      double accuracy = evaluate(testImages, testLabels);

      // Ausgabe von Verlust und Genauigkeit
      System.out.println(
          "Epoch " + (epoch + 1) + ": Loss = " + averageLoss + ", Accuracy = " + accuracy + "%");
    }
  }

  private void backward(Matrix3D errorGradient, double learningRate) {
    for (int i = layers.length - 1; i >= 0; i--) {
      errorGradient = layers[i].backward(errorGradient, learningRate);
    }
  }
}
