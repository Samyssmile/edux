package de.edux.ml.cnn.activation;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;

public class ReLU implements ActivationFunction {
    
    @Override
    public Tensor apply(Tensor x) {
        int[] shape = x.getShape();
        FloatTensor result = TensorPool.get(shape);
        
        float[] inputData = ((FloatTensor) x).getPrimitiveData();
        float[] outputData = result.getPrimitiveData();
        
        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = Math.max(0.0f, inputData[i]);
        }
        
        result.syncFromPrimitive();
        return result;
    }
    
    @Override
    public Tensor gradient(Tensor x) {
        int[] shape = x.getShape();
        FloatTensor result = TensorPool.get(shape);
        
        float[] inputData = ((FloatTensor) x).getPrimitiveData();
        float[] outputData = result.getPrimitiveData();
        
        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = inputData[i] > 0.0f ? 1.0f : 0.0f;
        }
        
        result.syncFromPrimitive();
        return result;
    }
}