package de.edux.ml.cnn.data;

import de.edux.ml.cnn.tensor.Tensor;

public class Batch {
    private final Tensor data;
    private final Tensor labels;
    
    public Batch(Tensor data, Tensor labels) {
        this.data = data;
        this.labels = labels;
    }
    
    public Tensor getData() {
        return data;
    }
    
    public Tensor getLabels() {
        return labels;
    }
    
    public int getBatchSize() {
        return data.getShape()[0];
    }
}