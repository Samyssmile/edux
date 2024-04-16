package de.edux.ml.mlp.util;

import de.edux.ml.api.core.tensor.Matrix;

public class TrainingMatrices {
  private Matrix input;
  private Matrix output;

  public TrainingMatrices(Matrix input, Matrix output) {
    this.input = input;
    this.output = output;
  }

  public Matrix getInput() {
    return input;
  }

  public void setInput(Matrix input) {
    this.input = input;
  }

  public Matrix getOutput() {
    return output;
  }

  public void setOutput(Matrix output) {
    this.output = output;
  }
}
