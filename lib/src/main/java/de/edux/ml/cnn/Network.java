package de.edux.ml.cnn;

import de.edux.ml.cnn.core.Tensor;
import de.edux.ml.cnn.layers.Layer;

import java.util.List;

public class Network extends Layer implements Trainable {

  private final List<Layer> layers;

  public Network(List<Layer> layers) {
    this.layers = layers;
  }

  @Override
  public Tensor forward(Tensor input) {
    for (Layer layer : layers) {
      input = layer.forward(input);
    }
    return input;
  }

  @Override
  public Tensor backward(Tensor input, double learningRate) {
    for (int i = layers.size() - 1; i >= 0; i--) {
      input = layers.get(i).backward(input, learningRate);
    }
    return input;
  }

  @Override
  public void train(Tensor inputs, Tensor targets, int epochs, int batchSize, double learningRate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      // Shuffling the training data might be a good idea here
      int totalBatches = inputs.getRows() / batchSize;
      double loss = 0.0;
      for (int batchNum = 0; batchNum < totalBatches; batchNum++) {
        int startIdx = batchNum * batchSize;
        int endIdx = Math.min(inputs.getRows(), startIdx + batchSize);

        // Extract the current batch
        Tensor batchInputs = inputs.getBatch(startIdx, endIdx);
        Tensor batchTargets = targets.getBatch(startIdx, endIdx);

        // Forward pass
        Tensor predictions = forward(batchInputs);

        // Calculate loss (Cross Entropy)
        loss = calculateCrossEntropyLoss(predictions, batchTargets);

        // Backward pass
        Tensor error =
            calculateError(predictions, batchTargets); // Implement this based on your loss function
        backward(error, learningRate);
      }

      // Optional: Print out loss and accuracy after each epoch
      System.out.println("Epoch " + epoch + ": Loss = " + loss);
    }
  }

  @Override
  public Tensor predict(Tensor input) {
    return null;
  }

  @Override
  public double evaluate(Tensor inputs, Tensor targets) {
    return 0;
  }

  private double calculateCrossEntropyLoss(Tensor predictions, Tensor targets) {
    double loss = 0.0;
    double[][] predData = predictions.getData();
    double[][] targetData = targets.getData();

    for (int i = 0; i < predictions.getRows(); i++) {
      for (int j = 0; j < predictions.getCols(); j++) {
        loss -=
            targetData[i][j]
                * Math.log(
                    Math.max(predData[i][j], 1e-15)); // Adding a small constant to prevent log(0)
      }
    }

    return loss / predictions.getRows(); // Average loss per data point
  }

  private Tensor calculateError(Tensor predictions, Tensor targets) {
    Tensor error = new Tensor(predictions.getRows(), predictions.getCols());
    double[][] predData = predictions.getData();
    double[][] targetData = targets.getData();
    double[][] errorData = error.getData();

    for (int i = 0; i < predictions.getRows(); i++) {
      for (int j = 0; j < predictions.getCols(); j++) {
        errorData[i][j] = predData[i][j] - targetData[i][j];
      }
    }

    return error;
  }
}
