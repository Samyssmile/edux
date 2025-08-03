package de.edux.ml.cnn.activation;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;

public class Softmax implements ActivationFunction {
    
    @Override
    public Tensor apply(Tensor x) {
        int[] shape = x.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Softmax expects 2D input: [batch, features]");
        }
        
        int batch = shape[0];
        int features = shape[1];
        
        FloatTensor result = FloatTensor.zeros(shape);
        float[] inputData = ((FloatTensor) x).getPrimitiveData();
        float[] outputData = result.getPrimitiveData();
        
        for (int b = 0; b < batch; b++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int f = 0; f < features; f++) {
                int idx = b * features + f;
                max = Math.max(max, inputData[idx]);
            }
            
            float sum = 0.0f;
            for (int f = 0; f < features; f++) {
                int idx = b * features + f;
                float exp = (float) Math.exp(inputData[idx] - max);
                outputData[idx] = exp;
                sum += exp;
            }
            
            for (int f = 0; f < features; f++) {
                int idx = b * features + f;
                outputData[idx] /= sum;
            }
        }
        
        // Sync primitive data back to boxed array
        result.syncFromPrimitive();
        
        return result;
    }
    
    @Override
    public Tensor gradient(Tensor x) {
        Tensor softmaxOutput = apply(x);
        int[] shape = softmaxOutput.getShape();
        FloatTensor result = FloatTensor.zeros(shape);
        
        Float[] softmaxData = (Float[]) softmaxOutput.getData();
        Float[] gradData = result.getDataArray();
        
        int batch = shape[0];
        int features = shape[1];
        
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < features; i++) {
                for (int j = 0; j < features; j++) {
                    int idx_i = b * features + i;
                    int idx_j = b * features + j;
                    
                    if (i == j) {
                        gradData[idx_i] += softmaxData[idx_i] * (1.0f - softmaxData[idx_i]);
                    } else {
                        gradData[idx_i] += -softmaxData[idx_i] * softmaxData[idx_j];
                    }
                }
            }
        }
        
        return result;
    }
}