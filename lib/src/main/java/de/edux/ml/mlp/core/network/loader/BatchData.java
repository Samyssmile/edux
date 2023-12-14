package de.edux.ml.mlp.core.network.loader;

public interface BatchData {

    double[] getInputBatch();

    void setInputBatch(double[] inputBatch);

    double[] getExpectedBatch();
    void setExpectedBatch(double[] expectedBatch);
}
