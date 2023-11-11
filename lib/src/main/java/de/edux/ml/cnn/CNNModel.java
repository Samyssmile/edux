package de.edux.ml.cnn;

import de.edux.ml.cnn.layers.Layer;
import de.edux.ml.cnn.loss.CrossEntropyLoss;
import de.edux.ml.cnn.math.Matrix;
import java.util.ArrayList;
import java.util.List;

public class CNNModel {
  private List<Layer> layers;
  private double learningRate;

  public CNNModel(double learningRate) {
    this.layers = new ArrayList<>();
    this.learningRate = learningRate;
  }

  public void addLayer(Layer layer) {
    layers.add(layer);
  }

  public Matrix forward(Matrix input) {
    Matrix output = input;
    for (Layer layer : layers) {
      output = layer.forward(output);
    }
    return output;
  }

  public void backward(Matrix outputGradient) {
    Matrix gradient = outputGradient;
    for (int i = layers.size() - 1; i >= 0; i--) {
      gradient = layers.get(i).backward(gradient, learningRate);
    }
  }

  public void train(
      List<Matrix> inputs,
      List<Matrix> targets,
      List<Matrix> testInputs,
      List<Matrix> testTargets,
      int epochs) {
    double loss = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
      for (int i = 0; i < inputs.size(); i++) {
        // Forward pass
        Matrix output = forward(inputs.get(i));
        Matrix target = convertTargetToOneHot(targets.get(i), targets.get(0).getRows());
        // Compute loss and gradient
        loss = CrossEntropyLoss.computeLoss(output, target);

        Matrix gradient = CrossEntropyLoss.computeGradient(output, target);

        // Backward pass and update weights
        backward(gradient);
      }
      double accuracy = calculateAccuracy(testInputs, testTargets);
      System.out.println(
          "Epoche " + epoch + " - Genauigkeit: " + accuracy + "%" + " - Loss: " + loss);
    }
  }

  public double calculateAccuracy(List<Matrix> testInputs, List<Matrix> testTargets) {
    int correctPredictions = 0;
    int totalPredictions = testInputs.size();

    for (int i = 0; i < totalPredictions; i++) {
      Matrix output = predict(testInputs.get(i));
      int predictedLabel = getLabelFromPrediction(output);
      int trueLabel = getLabelFromOneHot(testTargets.get(i));

      if (predictedLabel == trueLabel) {
        correctPredictions++;
      }
    }

    return (double) correctPredictions / totalPredictions * 100.0; // Umwandlung in Prozent
  }

  private Matrix convertTargetToOneHot(Matrix target, int numClasses) {
    // Hier wird angenommen, dass jede 'target'-Matrix genau eine Zeile und eine Spalte hat,
    // und der Wert in dieser Zelle den Klassenindex darstellt.
    int classIndex = (int) target.getData()[0][0];

    // Erstellen Sie einen One-Hot-Vektor.
    double[][] oneHotData = new double[1][numClasses];
    for (int i = 0; i < numClasses; i++) {
      oneHotData[0][i] = (i == classIndex) ? 1.0 : 0.0;
    }

    return new Matrix(oneHotData, 1, numClasses);
  }

  public double evaluate(List<Matrix> testImages, List<Matrix> testLabels) {
    int correctPredictions = 0;

    for (int i = 0; i < testImages.size(); i++) {
      Matrix prediction = this.predict(testImages.get(i));

      int predictedLabel = getLabelFromPrediction(prediction);
      int trueLabel = getLabelFromOneHot(testLabels.get(i));

      if (predictedLabel == trueLabel) {
        correctPredictions++;
      }
    }

    return correctPredictions / (double) testImages.size();
  }

  private int getLabelFromPrediction(Matrix prediction) {
    double maxProb = 0;
    int labelIndex = -1;

    for (int i = 0; i < prediction.getRows(); i++) {
      if (prediction.getData()[i][0] > maxProb) {
        maxProb = prediction.getData()[i][0];
        labelIndex = i;
      }
    }

    return labelIndex;
  }

  private int getLabelFromOneHot(Matrix oneHot) {
    for (int i = 0; i < oneHot.getRows(); i++) {
      if (oneHot.getData()[i][0] == 1.0) {
        return i;
      }
    }

    return -1; // Return -1 or throw an error if no '1' is found
  }

  public Matrix predict(Matrix input) {
    return forward(input);
  }
}
