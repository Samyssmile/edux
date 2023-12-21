package de.edux.ml.mlp.core.network;

import de.edux.ml.mlp.core.network.loss.LossFunction;
import de.edux.ml.mlp.core.network.loss.LossFunctions;
import de.edux.ml.mlp.core.tensor.Matrix;
import de.edux.ml.mlp.exceptions.UnsupportedLossFunction;
import java.io.Serializable;
import java.util.LinkedList;

// Network
public class Engine implements Layer, Serializable {
  private static final long serialVersionUID = 1L;
  private final LinkedList<Double> lossHistory = new LinkedList<>();
  private final LinkedList<Double> accuracyHistory = new LinkedList<>();

  private final LinkedList<Layer> layers = new LinkedList<>();

  private final LossFunction lossFunction = LossFunction.CROSS_ENTROPY;

  private transient RunningAverages runningAverages;
  private int batchSize;

  public Engine(int batchSize) {
    this.batchSize = batchSize;
    initAverageMetrics();
  }

  @Override
  public Matrix backwardLayerBased(Matrix error, float learningRate) {
    for (int i = layers.size() - 1; i >= 0; i--) {
      error = layers.get(i).backwardLayerBased(error, learningRate);
    }

    return error;
  }

  @Override
  public Matrix forwardLayerbased(Matrix input) {
    Matrix output = input;
    for (Layer layer : layers) {
      output = layer.forwardLayerbased(output);
    }
    return output;
  }

  public synchronized double evaluateLayerBased(Matrix predicted, Matrix expected) {
    if (LossFunction.CROSS_ENTROPY != lossFunction) {
      throw new UnsupportedLossFunction("Only Cross Entropy is supported.");
    }

    double loss = LossFunctions.crossEntropy(expected, predicted).averageColumn().get(0);
    Matrix predictions = predicted.getGreatestRowNumber();
    Matrix actual = expected.getGreatestRowNumber();

    int correct = 0;
    for (int i = 0; i < actual.getCols(); i++) {
      if (predictions.get(i) == actual.get(i)) {
        correct++;
      }
    }

    double percentCorrect = (100.0 * correct) / actual.getCols();
    this.accuracyHistory.add(percentCorrect);
    this.lossHistory.add(loss);
    if (this.runningAverages == null) {
      initAverageMetrics();
    }
    this.runningAverages.add(loss, percentCorrect);
    return loss;
  }

  private void initAverageMetrics() {
    this.runningAverages =
        new RunningAverages(
            2,
            this.batchSize,
            (callNumber, averages) -> {
              System.out.printf(
                  "Epoch: %d, Loss: %.2f, Accuracy: %.2f\n", callNumber, averages[0], averages[1]);
            });
  }

  public void addLayer(Layer layer) {
    layers.add(layer);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (Layer layer : layers) {
      sb.append(layer.toString());
      sb.append("\n");
    }
    return sb.toString();
  }

  public LinkedList<Double> getLossHistory() {
    return lossHistory;
  }

  public LinkedList<Double> getAccuracyHistory() {
    return accuracyHistory;
  }

  @Override
  public void updateWeightsAndBias() {
    for (Layer layer : layers) {
      layer.updateWeightsAndBias();
    }
  }

  public int getBatchSize() {
    return batchSize;
  }

  public void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }
}
