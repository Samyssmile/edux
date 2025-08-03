package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import java.util.stream.IntStream;
import java.io.Serializable;

public class PoolingLayer implements Layer, Serializable {
    public enum PoolingType {
        MAX, AVERAGE
    }
    
    private final PoolingType poolingType;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private boolean training = true;
    private Tensor lastInput;
    private int[][] maxIndices;
    private FloatTensor cachedGradInput;
    private FloatTensor cachedOutput;
    
    public PoolingLayer(PoolingType poolingType, int kernelSize, int stride, int padding) {
        this.poolingType = poolingType;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }
    
    public PoolingLayer(PoolingType poolingType, int kernelSize) {
        this(poolingType, kernelSize, kernelSize, 0);
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
        
        int outputHeight = (height + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (width + 2 * padding - kernelSize) / stride + 1;
        
        int[] outputShape = new int[]{batch, channels, outputHeight, outputWidth};
        
        // Reuse cached output tensor or get new one from pool
        if (cachedOutput == null || !java.util.Arrays.equals(cachedOutput.getShape(), outputShape)) {
            if (cachedOutput != null) {
                TensorPool.release(cachedOutput);
            }
            cachedOutput = TensorPool.get(outputShape);
        }
        
        FloatTensor output = cachedOutput;
        // Clear the output tensor
        float[] outputData = output.getPrimitiveData();
        java.util.Arrays.fill(outputData, 0.0f);
        
        if (poolingType == PoolingType.MAX) {
            maxIndices = new int[batch * channels * outputHeight * outputWidth][2];
        }
        
        FloatTensor inputTensor = (FloatTensor) input;
        float[] inputData = inputTensor.getPrimitiveData();
        
        IntStream.range(0, batch).parallel().forEach(b -> {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float result = 0.0f;
                        int maxH = -1, maxW = -1;
                        boolean first = true;
                        int poolCount = 0;
                        
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int inputIdx = b * channels * height * width + 
                                                  c * height * width + 
                                                  ih * width + iw;
                                    float value = inputData[inputIdx];
                                    
                                    if (poolingType == PoolingType.MAX) {
                                        if (first || value > result) {
                                            result = value;
                                            maxH = ih;
                                            maxW = iw;
                                            first = false;
                                        }
                                    } else if (poolingType == PoolingType.AVERAGE) {
                                        result += value;
                                        poolCount++;
                                    }
                                }
                            }
                        }
                        
                        if (poolingType == PoolingType.AVERAGE && poolCount > 0) {
                            result /= poolCount;
                        }
                        
                        int outputIdx = b * channels * outputHeight * outputWidth + 
                                       c * outputHeight * outputWidth + 
                                       oh * outputWidth + ow;
                        outputData[outputIdx] = result;
                        
                        if (poolingType == PoolingType.MAX) {
                            maxIndices[outputIdx][0] = maxH;
                            maxIndices[outputIdx][1] = maxW;
                        }
                    }
                }
            }
        });
        
        // Sync primitive data back to boxed array
        output.syncFromPrimitive();
        
        return output;
    }
    
    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward before backward");
        }
        
        int[] inputShape = lastInput.getShape();
        int[] gradOutputShape = gradOutput.getShape();
        
        // Reuse cached gradient tensor or get new one from pool
        if (cachedGradInput == null || !java.util.Arrays.equals(cachedGradInput.getShape(), inputShape)) {
            if (cachedGradInput != null) {
                TensorPool.release(cachedGradInput);
            }
            cachedGradInput = TensorPool.get(inputShape);
        }
        
        FloatTensor gradInput = cachedGradInput;
        // Clear the gradient tensor
        float[] gradInputData = gradInput.getPrimitiveData();
        java.util.Arrays.fill(gradInputData, 0.0f);
        
        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        int outputHeight = gradOutputShape[2];
        int outputWidth = gradOutputShape[3];
        
        FloatTensor gradOutputTensor = (FloatTensor) gradOutput;
        float[] gradOutputData = gradOutputTensor.getPrimitiveData();
        
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        int outputIdx = b * channels * outputHeight * outputWidth + 
                                       c * outputHeight * outputWidth + 
                                       oh * outputWidth + ow;
                        float gradOut = gradOutputData[outputIdx];
                        
                        if (poolingType == PoolingType.MAX) {
                            // For max pooling, gradient goes only to the max element
                            if (maxIndices != null && outputIdx < maxIndices.length) {
                                int maxH = maxIndices[outputIdx][0];
                                int maxW = maxIndices[outputIdx][1];
                                
                                if (maxH >= 0 && maxW >= 0) {
                                    int inputIdx = b * channels * height * width + 
                                                  c * height * width + 
                                                  maxH * width + maxW;
                                    gradInputData[inputIdx] += gradOut;
                                }
                            }
                        } else if (poolingType == PoolingType.AVERAGE) {
                            // For average pooling, gradient is distributed evenly
                            int poolCount = 0;
                            // First count valid positions
                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        poolCount++;
                                    }
                                }
                            }
                            
                            // Then distribute gradient
                            if (poolCount > 0) {
                                float gradPerElement = gradOut / poolCount;
                                for (int kh = 0; kh < kernelSize; kh++) {
                                    for (int kw = 0; kw < kernelSize; kw++) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        
                                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                            int inputIdx = b * channels * height * width + 
                                                          c * height * width + 
                                                          ih * width + iw;
                                            gradInputData[inputIdx] += gradPerElement;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Sync primitive data back to boxed array
        gradInput.syncFromPrimitive();
        
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
    
    @Override
    public void cleanup() {
        maxIndices = null;
        if (cachedGradInput != null) {
            TensorPool.release(cachedGradInput);
            cachedGradInput = null;
        }
        if (cachedOutput != null) {
            TensorPool.release(cachedOutput);
            cachedOutput = null;
        }
    }
}