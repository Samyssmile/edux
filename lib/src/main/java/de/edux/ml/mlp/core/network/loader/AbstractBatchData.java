package de.edux.ml.mlp.core.network.loader;

public abstract class AbstractBatchData implements BatchData {

    private double[] inputBatch;
    private double[] expectedBatch;

    @Override
    public double[] getInputBatch() {
        return inputBatch;
    }

    @Override
    public void setInputBatch(double[] inputBatch) {
        this.inputBatch = inputBatch;
    }

    @Override
    public double[] getExpectedBatch() {
        return expectedBatch;
    }

    @Override
    public void setExpectedBatch(double[] expectedBatch) {
        this.expectedBatch = expectedBatch;
    }
}
