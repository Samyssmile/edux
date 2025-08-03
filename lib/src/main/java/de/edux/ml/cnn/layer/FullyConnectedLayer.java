package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import java.io.Serializable;

public class FullyConnectedLayer implements Layer, Serializable {
    private final int inputSize;
    private final int outputSize;
    private final boolean bias;
    
    private FloatTensor weights;
    private FloatTensor biases;
    private boolean training = true;
    private Tensor lastInput;
    private FloatTensor weightGradients;
    private FloatTensor biasGradients;
    
    public FullyConnectedLayer(int inputSize, int outputSize, boolean bias) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.bias = bias;
        
        initializeWeights();
    }
    
    public FullyConnectedLayer(int inputSize, int outputSize) {
        this(inputSize, outputSize, true);
    }
    
    private void initializeWeights() {
        int[] weightShape = new int[]{inputSize, outputSize};
        weights = FloatTensor.zeros(weightShape);
        
        float std = (float) Math.sqrt(2.0 / inputSize);
        float[] weightData = weights.getPrimitiveData();
        for (int i = 0; i < weightData.length; i++) {
            weightData[i] = (float) (std * (Math.random() * 2 - 1));
        }
        weights.syncFromPrimitive();
        
        if (bias) {
            biases = FloatTensor.zeros(outputSize);
            float[] biasData = biases.getPrimitiveData();
            for (int i = 0; i < biasData.length; i++) {
                biasData[i] = (float) (0.01 * (Math.random() * 2 - 1));
            }
            biases.syncFromPrimitive();
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        
        int[] inputShape = input.getShape();
        
        if (inputShape.length != 2) {
            throw new IllegalArgumentException("Input must be 2D: [batch, features]");
        }
        
        int batch = inputShape[0];
        int features = inputShape[1];
        
        if (features != inputSize) {
            throw new IllegalArgumentException("Input features don't match layer configuration");
        }
        
        FloatTensor inputTensor = (FloatTensor) input;
        FloatTensor output = inputTensor.matmul(weights);
        
        if (bias) {
            // Optimized bias addition using primitive arrays
            float[] outputData = output.getPrimitiveData();
            float[] biasData = biases.getPrimitiveData();
            
            for (int b = 0; b < batch; b++) {
                for (int o = 0; o < outputSize; o++) {
                    int idx = b * outputSize + o;
                    outputData[idx] += biasData[o];
                }
            }
            output.syncFromPrimitive();
        }
        
        return output;
    }
    
    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward before backward");
        }
        
        // gradOutput shape: [batch, outputSize]
        // lastInput shape: [batch, inputSize]  
        // weights shape: [inputSize, outputSize]
        
        int[] inputShape = lastInput.getShape();
        int batchSize = inputShape[0];
        
        FloatTensor gradOutputTensor = (FloatTensor) gradOutput;
        FloatTensor inputTensor = (FloatTensor) lastInput;
        
        
        // Compute gradient w.r.t. weights: input.T @ gradOutput
        FloatTensor inputTransposed = inputTensor.transpose();
        FloatTensor weightGradient = inputTransposed.matmul(gradOutputTensor);
        
        
        // Compute gradient w.r.t. input: gradOutput @ weights.T  
        FloatTensor weightsTransposed = weights.transpose();
        FloatTensor gradInput = gradOutputTensor.matmul(weightsTransposed);
        
        // Store gradients (accumulate for batch)
        if (weightGradients == null) {
            weightGradients = weightGradient;
        } else {
            weightGradients.addInPlace(weightGradient);
            TensorPool.release(weightGradient); // Release temporary tensor
        }
        
        if (bias && biasGradients == null) {
            // Bias gradient is sum over batch dimension
            biasGradients = gradOutputTensor.sum(0);
        } else if (bias) {
            FloatTensor biasGrad = gradOutputTensor.sum(0);
            biasGradients.addInPlace(biasGrad);
            TensorPool.release(biasGrad); // Release temporary tensor
        }
        
        // Release temporary tensors
        TensorPool.release(inputTransposed);
        TensorPool.release(weightsTransposed);
        
        
        return gradInput;
    }
    
    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    @Override
    public boolean isTraining() {
        return training;
    }
    
    public FloatTensor getWeights() {
        return weights;
    }
    
    public FloatTensor getBiases() {
        return biases;
    }
    
    public FloatTensor getWeightGradients() {
        return weightGradients;
    }
    
    public FloatTensor getBiasGradients() {
        return biasGradients;
    }
    
    public void zeroGradients() {
        if (weightGradients != null) {
            TensorPool.release(weightGradients);
            weightGradients = null;
        }
        if (biasGradients != null) {
            TensorPool.release(biasGradients);
            biasGradients = null;
        }
    }
    
    @Override
    public void cleanup() {
        zeroGradients();
    }
}