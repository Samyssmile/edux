package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import de.edux.ml.cnn.tensor.Conv2DUtils;
import java.util.stream.IntStream;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.io.Serializable;

public class ConvolutionalLayer implements Layer, Serializable {
    private final int inputChannels;
    private final int outputChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final boolean bias;
    
    private FloatTensor weights;
    private FloatTensor biases;
    private boolean training = true;
    private Tensor lastInput;
    private FloatTensor weightGradients;
    private FloatTensor biasGradients;
    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();
    
    // Cached tensors for reuse
    private FloatTensor cachedOutput;
    private FloatTensor cachedGradInput;
    
    public ConvolutionalLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding, boolean bias) {
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.bias = bias;
        
        initializeWeights();
    }
    
    public ConvolutionalLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding) {
        this(inputChannels, outputChannels, kernelSize, stride, padding, true);
    }
    
    public ConvolutionalLayer(int inputChannels, int outputChannels, int kernelSize) {
        this(inputChannels, outputChannels, kernelSize, 1, 0, true);
    }
    
    private void initializeWeights() {
        int[] weightShape = new int[]{outputChannels, inputChannels, kernelSize, kernelSize};
        weights = FloatTensor.zeros(weightShape);
        
        float std = (float) Math.sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        float[] weightData = weights.getPrimitiveData();
        for (int i = 0; i < weightData.length; i++) {
            weightData[i] = (float) (std * (Math.random() * 2 - 1));
        }
        weights.syncFromPrimitive();
        
        if (bias) {
            biases = FloatTensor.zeros(outputChannels);
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
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Input must be 4D: [batch, channels, height, width]");
        }
        
        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        
        if (channels != inputChannels) {
            throw new IllegalArgumentException("Input channels don't match layer configuration");
        }
        
        int outputHeight = (height + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (width + 2 * padding - kernelSize) / stride + 1;
        
        // Use optimized convolution with tensor pooling
        FloatTensor inputTensor = (FloatTensor) input;
        
        
        FloatTensor output = Conv2DUtils.conv2d(inputTensor, weights, bias ? biases : null, stride, padding);
        
        
        // Cache for backward pass
        if (cachedOutput != null) {
            TensorPool.release(cachedOutput);
        }
        cachedOutput = output;
        
        return output;
    }
    
    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward before backward");
        }
        
        int[] inputShape = lastInput.getShape();
        int[] gradOutputShape = gradOutput.getShape();
        
        int batch = inputShape[0];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        int outputHeight = gradOutputShape[2];
        int outputWidth = gradOutputShape[3];
        
        // Use tensor pooling for gradient tensors
        FloatTensor gradInput = TensorPool.get(inputShape);
        
        FloatTensor inputTensor = (FloatTensor) lastInput;
        FloatTensor gradOutputTensor = (FloatTensor) gradOutput;
        
        // Initialize or reset weight gradients
        if (weightGradients == null) {
            weightGradients = TensorPool.get(weights.getShape());
        } else {
            float[] data = weightGradients.getPrimitiveData();
            for (int i = 0; i < data.length; i++) {
                data[i] = 0.0f;
            }
            weightGradients.syncFromPrimitive();
        }
        
        // Initialize or reset bias gradients
        if (bias) {
            if (biasGradients == null) {
                biasGradients = TensorPool.get(biases.getShape());
            } else {
                float[] data = biasGradients.getPrimitiveData();
                for (int i = 0; i < data.length; i++) {
                    data[i] = 0.0f;
                }
                biasGradients.syncFromPrimitive();
            }
        }
        
        // Parallel gradient computation
        forkJoinPool.invoke(new BackwardTask(
            inputTensor.getPrimitiveData(),
            gradOutputTensor.getPrimitiveData(),
            gradInput.getPrimitiveData(),
            weights.getPrimitiveData(),
            weightGradients.getPrimitiveData(),
            bias ? biasGradients.getPrimitiveData() : null,
            batch, inputChannels, inputHeight, inputWidth,
            outputChannels, outputHeight, outputWidth,
            kernelSize, stride, padding,
            0, batch
        ));
        
        // Sync primitive arrays back to boxed arrays
        gradInput.syncFromPrimitive();
        weightGradients.syncFromPrimitive();
        if (bias && biasGradients != null) {
            biasGradients.syncFromPrimitive();
        }
        
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
        if (cachedOutput != null) {
            TensorPool.release(cachedOutput);
            cachedOutput = null;
        }
        if (cachedGradInput != null) {
            TensorPool.release(cachedGradInput);
            cachedGradInput = null;
        }
        zeroGradients();
    }
    
    /**
     * Recursive task for parallel backward computation
     */
    private static class BackwardTask extends RecursiveAction {
        private static final int THRESHOLD = 4;
        
        private final float[] input, gradOutput, gradInput, weights, weightGrads, biasGrads;
        private final int batch, inputChannels, inputHeight, inputWidth;
        private final int outputChannels, outputHeight, outputWidth;
        private final int kernelSize, stride, padding;
        private final int startBatch, endBatch;
        
        BackwardTask(float[] input, float[] gradOutput, float[] gradInput,
                    float[] weights, float[] weightGrads, float[] biasGrads,
                    int batch, int inputChannels, int inputHeight, int inputWidth,
                    int outputChannels, int outputHeight, int outputWidth,
                    int kernelSize, int stride, int padding,
                    int startBatch, int endBatch) {
            this.input = input;
            this.gradOutput = gradOutput;
            this.gradInput = gradInput;
            this.weights = weights;
            this.weightGrads = weightGrads;
            this.biasGrads = biasGrads;
            this.batch = batch;
            this.inputChannels = inputChannels;
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
            this.outputChannels = outputChannels;
            this.outputHeight = outputHeight;
            this.outputWidth = outputWidth;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
            this.startBatch = startBatch;
            this.endBatch = endBatch;
        }
        
        @Override
        protected void compute() {
            if (endBatch - startBatch <= THRESHOLD) {
                computeDirectly();
            } else {
                int mid = (startBatch + endBatch) / 2;
                invokeAll(
                    new BackwardTask(input, gradOutput, gradInput, weights, weightGrads, biasGrads,
                                   batch, inputChannels, inputHeight, inputWidth,
                                   outputChannels, outputHeight, outputWidth,
                                   kernelSize, stride, padding, startBatch, mid),
                    new BackwardTask(input, gradOutput, gradInput, weights, weightGrads, biasGrads,
                                   batch, inputChannels, inputHeight, inputWidth,
                                   outputChannels, outputHeight, outputWidth,
                                   kernelSize, stride, padding, mid, endBatch)
                );
            }
        }
        
        private void computeDirectly() {
            int inputBatchStride = inputChannels * inputHeight * inputWidth;
            int outputBatchStride = outputChannels * outputHeight * outputWidth;
            int weightFilterStride = inputChannels * kernelSize * kernelSize;
            
            for (int b = startBatch; b < endBatch; b++) {
                int inputBatchOffset = b * inputBatchStride;
                int outputBatchOffset = b * outputBatchStride;
                
                for (int oc = 0; oc < outputChannels; oc++) {
                    for (int oh = 0; oh < outputHeight; oh++) {
                        for (int ow = 0; ow < outputWidth; ow++) {
                            int gradOutputIdx = outputBatchOffset + oc * outputHeight * outputWidth + oh * outputWidth + ow;
                            float gradOut = gradOutput[gradOutputIdx];
                            
                            // Compute bias gradients (synchronized)
                            if (biasGrads != null) {
                                synchronized (biasGrads) {
                                    biasGrads[oc] += gradOut;
                                }
                            }
                            
                            // Compute weight gradients and input gradients
                            for (int ic = 0; ic < inputChannels; ic++) {
                                for (int kh = 0; kh < kernelSize; kh++) {
                                    for (int kw = 0; kw < kernelSize; kw++) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        
                                        if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                            int inputIdx = inputBatchOffset + ic * inputHeight * inputWidth + ih * inputWidth + iw;
                                            int weightIdx = oc * weightFilterStride + ic * kernelSize * kernelSize + kh * kernelSize + kw;
                                            
                                            // Weight gradient: input * gradOutput (synchronized)
                                            synchronized (weightGrads) {
                                                weightGrads[weightIdx] += input[inputIdx] * gradOut;
                                            }
                                            
                                            // Input gradient: weight * gradOutput
                                            gradInput[inputIdx] += weights[weightIdx] * gradOut;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}