package de.edux.ml.mlp.core.network;

import de.edux.ml.mlp.core.tensor.Matrix;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class BatchResult implements Serializable {

  private AtomicReference<Matrix> accumulatedWeightGradient = new AtomicReference<>();
  private AtomicReference<Matrix> accumulatedBiasGradient = new AtomicReference<>();
  private AtomicReference<Matrix> lastInput = new AtomicReference<>();

  private AtomicReference<Float> learningRate = new AtomicReference<>();

  private AtomicInteger counter = new AtomicInteger(0);

  private AtomicInteger length = new AtomicInteger(0);

  public BatchResult() {
    counter.incrementAndGet();
  }

  public AtomicInteger getCounter() {
    return counter;
  }

  public synchronized void addGradients(
      Matrix weightsGradient, Matrix biasGradient, float learningRate, Matrix lastInput) {

    this.length.incrementAndGet();

    if (accumulatedWeightGradient.get() == null) {
      accumulatedWeightGradient.set(weightsGradient);
    } else {
      accumulatedWeightGradient.set(accumulatedWeightGradient.get().add(weightsGradient));
    }

    if (accumulatedBiasGradient.get() == null) {
      accumulatedBiasGradient.set(biasGradient);
    } else {
      accumulatedBiasGradient.set(accumulatedBiasGradient.get().add(biasGradient));
    }

    this.lastInput.set(lastInput);
    this.learningRate.set(learningRate);
  }

  public AtomicReference<Matrix> getAccumulatedWeightGradient() {
    return accumulatedWeightGradient;
  }

  public AtomicReference<Matrix> getAccumulatedBiasGradient() {
    return accumulatedBiasGradient;
  }

  public AtomicReference<Matrix> getLastInput() {
    return lastInput;
  }

  public AtomicReference<Float> getLearningRate() {
    return learningRate;
  }

  public void clear() {
    accumulatedWeightGradient.set(null);
    accumulatedBiasGradient.set(null);
    lastInput.set(null);
    learningRate.set(null);
    length.set(0);
  }

  public int getLength() {
    return length.get();
  }
}
