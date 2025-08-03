package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.activation.ReLU;
import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import java.io.Serializable;

public class ReLuLayer implements Layer, Serializable {
    private final ReLU activation;
    private boolean training = true;
    private Tensor lastInput;
    private FloatTensor cachedGradient;
    private FloatTensor cachedOutput;
    
    public ReLuLayer() {
        this.activation = new ReLU();
    }
    
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        
        int[] inputShape = input.getShape();
        if (cachedOutput == null || !java.util.Arrays.equals(cachedOutput.getShape(), inputShape)) {
            if (cachedOutput != null) {
                TensorPool.release(cachedOutput);
            }
            cachedOutput = TensorPool.get(inputShape);
        }
        
        FloatTensor inputTensor = (FloatTensor) input;
        float[] inputData = inputTensor.getPrimitiveData();
        float[] outputData = cachedOutput.getPrimitiveData();
        
        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = Math.max(0.0f, inputData[i]);
        }
        
        cachedOutput.syncFromPrimitive();
        return cachedOutput;
    }
    
    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward before backward");
        }
        
        int[] inputShape = lastInput.getShape();
        if (cachedGradient == null || !java.util.Arrays.equals(cachedGradient.getShape(), inputShape)) {
            if (cachedGradient != null) {
                TensorPool.release(cachedGradient);
            }
            cachedGradient = TensorPool.get(inputShape);
        }
        
        FloatTensor inputTensor = (FloatTensor) lastInput;
        FloatTensor gradOutputTensor = (FloatTensor) gradOutput;
        
        float[] inputData = inputTensor.getPrimitiveData();
        float[] gradOutputData = gradOutputTensor.getPrimitiveData();
        float[] resultData = cachedGradient.getPrimitiveData();
        
        for (int i = 0; i < inputData.length; i++) {
            resultData[i] = (inputData[i] > 0.0f) ? gradOutputData[i] : 0.0f;
        }
        
        cachedGradient.syncFromPrimitive();
        return cachedGradient;
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
        if (cachedGradient != null) {
            TensorPool.release(cachedGradient);
            cachedGradient = null;
        }
        if (cachedOutput != null) {
            TensorPool.release(cachedOutput);
            cachedOutput = null;
        }
    }
}