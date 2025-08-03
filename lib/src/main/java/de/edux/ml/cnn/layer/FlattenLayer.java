package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import java.io.Serializable;

public class FlattenLayer implements Layer, Serializable {
    private boolean training = true;
    private int[] originalShape;
    private FloatTensor cachedOutput;
    private FloatTensor cachedGradInput;
    
    @Override
    public Tensor forward(Tensor input) {
        originalShape = input.getShape();
        
        if (originalShape.length < 2) {
            throw new IllegalArgumentException("Input must have at least 2 dimensions");
        }
        
        int batch = originalShape[0];
        int totalFeatures = 1;
        for (int i = 1; i < originalShape.length; i++) {
            totalFeatures *= originalShape[i];
        }
        
        FloatTensor inputTensor = (FloatTensor) input;
        
        // Use cached output tensor or get new one from pool
        int[] outputShape = new int[]{batch, totalFeatures};
        if (cachedOutput == null || !java.util.Arrays.equals(cachedOutput.getShape(), outputShape)) {
            if (cachedOutput != null) {
                TensorPool.release(cachedOutput);
            }
            cachedOutput = TensorPool.get(outputShape);
        }
        
        // Copy data to cached output
        System.arraycopy(inputTensor.getPrimitiveData(), 0, cachedOutput.getPrimitiveData(), 0, inputTensor.getPrimitiveData().length);
        cachedOutput.syncFromPrimitive();
        
        Tensor result = cachedOutput;
        
        
        return result;
    }
    
    @Override
    public Tensor backward(Tensor gradOutput) {
        if (originalShape == null) {
            throw new IllegalStateException("Must call forward before backward");
        }
        
        // Use cached gradient input tensor or get new one from pool
        if (cachedGradInput == null || !java.util.Arrays.equals(cachedGradInput.getShape(), originalShape)) {
            if (cachedGradInput != null) {
                TensorPool.release(cachedGradInput);
            }
            cachedGradInput = TensorPool.get(originalShape);
        }
        
        // Copy data to cached gradient input
        FloatTensor gradOutputTensor = (FloatTensor) gradOutput;
        System.arraycopy(gradOutputTensor.getPrimitiveData(), 0, cachedGradInput.getPrimitiveData(), 0, gradOutputTensor.getPrimitiveData().length);
        cachedGradInput.syncFromPrimitive();
        
        return cachedGradInput;
    }
    
    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    @Override
    public boolean isTraining() {
        return training;
    }
    
    @Override
    public void cleanup() {
        originalShape = null;
        if (cachedOutput != null) {
            TensorPool.release(cachedOutput);
            cachedOutput = null;
        }
        if (cachedGradInput != null) {
            TensorPool.release(cachedGradInput);
            cachedGradInput = null;
        }
    }
}