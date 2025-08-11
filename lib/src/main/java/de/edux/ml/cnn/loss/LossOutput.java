package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.tensor.Tensor;

public class LossOutput {
    private final float loss;
    private final Tensor gradient;

    public LossOutput(float loss, Tensor gradient) {
        this.loss = loss;
        this.gradient = gradient;
    }

    public float getLoss() {
        return loss;
    }
    
    public Tensor getGradient() {
        return gradient;
    }
}