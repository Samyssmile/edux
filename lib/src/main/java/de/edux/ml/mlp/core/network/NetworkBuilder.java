package de.edux.ml.mlp.core.network;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
  private int batchSize = 100;
  private float initialLearningRate = 0.05f;
  private float finalLearningRate = 0.001f;

  private int threads = 1;
  private int epochs = 10;

  private List<Layer> layers = new ArrayList<>();

  public NetworkBuilder addLayer(Layer layer) {
    layers.add(layer);
    return this;
  }

  public NetworkBuilder withBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  public NetworkBuilder withLearningRates(float initialLearningRate, float finalLearningRate) {
    this.initialLearningRate = initialLearningRate;
    this.finalLearningRate = finalLearningRate;
    return this;
  }

  public NetworkBuilder withThreads(int threads) {
    this.threads = threads;
    return this;
  }

  public NetworkBuilder withEpochs(int epochs) {
    this.epochs = epochs;
    return this;
  }

  public NeuralNetwork build() {
    NeuralNetwork nn = new NeuralNetwork(batchSize);
    nn.setLearningRates(initialLearningRate, finalLearningRate);
    nn.setEpochs(epochs);
    nn.setThreads(threads);
    for (Layer layer : layers) {
      nn.addLayer(layer);
    }
    return nn;
  }

  public NeuralNetwork loadModel(String modelname) {
    NeuralNetwork nn = NeuralNetwork.loadModel(modelname);
    nn.setLearningRates(initialLearningRate, finalLearningRate);
    nn.setEpochs(epochs);
    nn.setThreads(threads);
    return nn;
  }
}
