package de.edux.ml.cnn;

import de.edux.ml.cnn.layers.*;
import de.edux.ml.cnn.loss.CrossEntropyLossV2;
import de.edux.ml.cnn.math.Matrix3D;
import java.util.List;

public class SimpleCNN {
  // Hilfsmethoden zum Hinzufügen und Teilen von Gradienten
  private static final double GRADIENT_CLIP_THRESHOLD = 1000.0; // Je nach Bedarf anpassen
  Matrix3D batchErrorGradient = null;
  private Layer[] layers;

  public SimpleCNN() {
    this.layers =
        new Layer[] {
          new ConvolutionalLayer(8, 3, 2, 1, 1),
          new ReLuLayer(),
          new FlattenLayer(),
          new DenseLayer(10, 1),
          new ReLuLayer()
        };
  }

  public Matrix3D forward(Matrix3D input) {
    Matrix3D output = input;
    for (Layer layer : layers) {
      output = layer.forward(output);
    }
    return output;
  }

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
    return (double) correctPredictions / testImages.size() * 100;
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
      int batchSize, // Neuer Parameter für die Batchgröße
      List<Matrix3D> testImages,
      List<Matrix3D> testLabels) {

    CrossEntropyLossV2 lossFunction = new CrossEntropyLossV2();

    for (int epoch = 0; epoch < epochs; epoch++) {
      double totalLoss = 0.0;
      batchErrorGradient = null;

      // Aufteilung der Trainingsdaten in Batches
      int numBatches = (int) Math.ceil((double) trainImages.size() / batchSize);
      for (int batch = 0; batch < numBatches; batch++) {
        int start = batch * batchSize;
        int end = Math.min(start + batchSize, trainImages.size());

        for (int i = start; i < end; i++) {
          Matrix3D input = trainImages.get(i);
          Matrix3D trueOutput = trainLabels.get(i);

          // Forward Pass
          Matrix3D predictedOutput = forward(input);

          // Berechnung des Verlustes
          double loss = lossFunction.computeLoss(predictedOutput, trueOutput);
          totalLoss += loss;

          // Berechnung des Gradienten des Verlustes und Akkumulation
          Matrix3D errorGradient = lossFunction.computeGradient(predictedOutput, trueOutput);

          if (batchErrorGradient == null) {
            batchErrorGradient =
                new Matrix3D(
                    errorGradient.getDepth(), errorGradient.getRows(), errorGradient.getCols());
          }

          batchErrorGradient =
              addGradients(batchErrorGradient, errorGradient); // Akkumulation der Gradienten
        }

        // Durchschnittlichen Gradienten berechnen
        Matrix3D averageGradient = divideGradient(batchErrorGradient, end - start);

        // Backward Pass für den gesamten Batch
        backward(averageGradient, learningRate);
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

  private Matrix3D addGradients(Matrix3D gradient1, Matrix3D gradient2) {
    if (gradient1.getDepth() != gradient2.getDepth()
        || gradient1.getRows() != gradient2.getRows()
        || gradient1.getCols() != gradient2.getCols()) {
      throw new IllegalArgumentException("Matrix dimensions must match for addition.");
    }

    Matrix3D result = new Matrix3D(gradient1.getDepth(), gradient1.getRows(), gradient1.getCols());
    for (int d = 0; d < result.getDepth(); d++) {
      for (int r = 0; r < result.getRows(); r++) {
        for (int c = 0; c < result.getCols(); c++) {
          double sum = gradient1.get(d, r, c) + gradient2.get(d, r, c);

          // Clipping der Gradientenwerte, falls sie zu groß sind
          if (sum > GRADIENT_CLIP_THRESHOLD) {
            sum = GRADIENT_CLIP_THRESHOLD;
          } else if (sum < -GRADIENT_CLIP_THRESHOLD) {
            sum = -GRADIENT_CLIP_THRESHOLD;
          }

          result.set(d, r, c, sum);
        }
      }
    }
    return result;
  }

  private Matrix3D divideGradient(Matrix3D gradient, int divisor) {
    Matrix3D result = new Matrix3D(gradient.getDepth(), gradient.getRows(), gradient.getCols());
    for (int d = 0; d < gradient.getDepth(); d++) {
      for (int r = 0; r < gradient.getRows(); r++) {
        for (int c = 0; c < gradient.getCols(); c++) {
          result.set(d, r, c, gradient.get(d, r, c));
        }
      }
    }
    result.divide(divisor);
    return result;
  }

  private void backward(Matrix3D errorGradient, double learningRate) {
    for (int i = layers.length - 1; i >= 0; i--) {
      errorGradient = layers[i].backward(errorGradient, learningRate);
    }
  }
}
