package de.edux.ml.mlp.core.network;

import de.edux.api.Classifier;
import de.edux.ml.mlp.core.network.loader.BatchData;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.core.tensor.Matrix;
import java.io.*;
import java.util.LinkedList;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NeuralNetwork implements Serializable {
  @Serial private static final long serialVersionUID = 1L;
  private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);
  private final Engine engine;
  private int epochs;
  private float initialLearningRate;
  private float finalLearningRate;
  private transient float learningRate;

  private int threads = 8;

  NeuralNetwork(int batchSize) {
    engine = new Engine(batchSize);
  }

  public void setLearningRates(float initialLearningRate, float finalLearningRate) {
    this.initialLearningRate = initialLearningRate;
    this.finalLearningRate = finalLearningRate;
  }

  public NeuralNetwork fit(Loader trainLoader, Loader evalLoader) {
    learningRate = initialLearningRate;
    for (int epoch = 0; epoch < epochs; epoch++) {
      runEpochLayerBased(trainLoader, true);

      if (evalLoader != null) {
        runEpochLayerBased(evalLoader, false);
      }

      learningRate -= (initialLearningRate - finalLearningRate) / epochs;
    }
    return this;
  }

  private void runEpochLayerBased(Loader loader, boolean traingMode) {
    loader.open();

    var queue = createBatchTasks(loader, traingMode);
    consumeBatchTasksLayerbased(queue, traingMode);

    loader.close();

    if (traingMode) {
      engine.updateWeightsAndBias();
    }
  }

  private Matrix runBatch(Loader loader, boolean trainingMode) {
    MetaData metaData = loader.getMetaData();
    BatchData batchData = loader.readBatch();
    int itemsRead = metaData.getItemsRead();
    int inputSize = metaData.getInputSize();
    int expectedSize = metaData.getExpectedSize();

    Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
    Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());

    Matrix batchResult = engine.forwardLayerbased(input);

    if (trainingMode) {
      engine.backwardLayerBased(expected, learningRate);
    } else {
      engine.evaluateLayerBased(batchResult, expected);
    }

    return batchResult;
  }

  private synchronized void consumeBatchTasksLayerbased(
      LinkedList<Future<Matrix>> batches, boolean traingMode) {
    int numberBatches = batches.size();
    int index = 0;

    for (var batch : batches) {
      try {
        var batchResult = batch.get();

      } catch (Exception e) {
        e.printStackTrace();
      }

      int printDot = (numberBatches / 25) + 1;
      if (traingMode && index++ % printDot == 0) {
        System.out.print(".");
      }
    }
  }

  private LinkedList<Future<Matrix>> createBatchTasks(Loader loader, boolean trainingMode) {
    LinkedList<Future<Matrix>> batches = new LinkedList<>();

    MetaData metaData = loader.getMetaData();
    var numberBatches = metaData.getNumberBatches();

    var executor = Executors.newFixedThreadPool(threads);

    for (int i = 0; i < numberBatches; i++) {
      batches.add(executor.submit(() -> runBatch(loader, trainingMode)));
    }

    executor.shutdown();

    return batches;
  }

  @Override
  public String toString() {
    return String.format("Neural Network Configuration\n")
        + "----------------------------------------\n"
        + String.format("Epochs: %d\n", epochs)
        + String.format("Batch size: %d\n", engine.getBatchSize())
        + String.format(
            "Initial learning rate: %f, Final learning rate: %f\n",
            initialLearningRate, finalLearningRate)
        + String.format("Threads: %d\n", threads)
        + "\nNetwork Architecture:"
        + "\n----------------------------------------\n"
        + engine.toString();
  }

  public void setBatchSize(int batchSize) {
    engine.setBatchSize(batchSize);
  }

  public boolean saveModel(String fileName) {
    File file = new File(fileName);
    try (var ds = new ObjectOutputStream(new FileOutputStream(file))) {
      ds.writeObject(this);
      log.info("Model saved to {}", file.getAbsolutePath());
    } catch (IOException e) {
      e.printStackTrace();
      return false;
    }

    return true;
  }

  public static NeuralNetwork loadModel(String fileName) {
    NeuralNetwork model = null;
    File file = new File(fileName);
    if (!file.exists()) {
      return null;
    }
    try (var ds = new ObjectInputStream(new FileInputStream(file))) {
      model = (NeuralNetwork) ds.readObject();
    } catch (IOException | ClassNotFoundException e) {
      e.printStackTrace();
    }
    log.info("Model loaded from {}", file.getAbsolutePath());
    return model;
  }

  public double[] predict(Matrix input) {
    return engine.forwardLayerbased(input).getData();
  }

  public LinkedList<Double> getLossHistory() {
    return engine.getLossHistory();
  }

  public LinkedList<Double> getAccuracyHistory() {
    return engine.getAccuracyHistory();
  }

  public void setThreads(int threads) {
    this.threads = threads;
  }

  public void addLayer(Layer layer) {
    engine.addLayer(layer);
  }

  public void setEpochs(int epochs) {
    this.epochs = epochs;
  }

  public NeuralNetwork printArchitecture() {
    System.out.println(this);
    return this;
  }
}
