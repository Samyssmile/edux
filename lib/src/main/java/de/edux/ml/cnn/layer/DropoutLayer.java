package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import java.io.Serializable;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ThreadLocalRandom;

/**
 * High-performance Dropout layer with parallel execution and optimized memory usage
 * Supports both training and inference modes with vectorized operations
 */
public class DropoutLayer implements Layer, Serializable {
    private final float dropoutRate;
    private final float keepRate;
    private final float invKeepRate;  // Pre-computed 1/keepRate for scaling
    
    // Training state
    private boolean training = true;
    private transient FloatTensor dropoutMask;  // Mask for backward pass
    
    // Performance optimization
    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();
    private static final int PARALLEL_THRESHOLD = 4096;  // Elements threshold for parallel execution
    
    // Thread-safe random number generation
    private static final ThreadLocal<Random> threadLocalRandom = ThreadLocal.withInitial(() -> new Random());
    
    public DropoutLayer(float dropoutRate) {
        if (dropoutRate < 0.0f || dropoutRate >= 1.0f) {
            throw new IllegalArgumentException("Dropout rate must be in [0, 1), got: " + dropoutRate);
        }
        
        this.dropoutRate = dropoutRate;
        this.keepRate = 1.0f - dropoutRate;
        this.invKeepRate = 1.0f / keepRate;
    }
    
    @Override
    public Tensor forward(Tensor input) {
        if (!training || dropoutRate == 0.0f) {
            // During inference or no dropout, return input unchanged
            return input;
        }
        
        int[] inputShape = input.getShape();
        FloatTensor inputTensor = (FloatTensor) input;
        FloatTensor output = TensorPool.get(inputShape);
        
        // Generate dropout mask
        if (dropoutMask != null) {
            TensorPool.release(dropoutMask);
        }
        dropoutMask = TensorPool.get(inputShape);
        
        int totalElements = input.size();
        float[] inputData = inputTensor.getPrimitiveData();
        float[] outputData = output.getPrimitiveData();
        float[] maskData = dropoutMask.getPrimitiveData();
        
        // Use parallel execution for large tensors
        if (totalElements > PARALLEL_THRESHOLD) {
            forkJoinPool.invoke(new DropoutForwardTask(inputData, outputData, maskData, 
                                                     keepRate, invKeepRate, 0, totalElements));
        } else {
            applyDropoutSequential(inputData, outputData, maskData, totalElements);
        }
        
        output.syncFromPrimitive();
        dropoutMask.syncFromPrimitive();
        
        return output;
    }
    
    private void applyDropoutSequential(float[] input, float[] output, float[] mask, int totalElements) {
        Random random = threadLocalRandom.get();
        
        // Vectorized dropout application with pre-computed scaling
        for (int i = 0; i < totalElements; i++) {
            if (random.nextFloat() < keepRate) {
                mask[i] = invKeepRate;  // Store inverse scaling factor
                output[i] = input[i] * invKeepRate;
            } else {
                mask[i] = 0.0f;
                output[i] = 0.0f;
            }
        }
    }
    
    @Override
    public Tensor backward(Tensor gradOutput) {
        if (!training || dropoutRate == 0.0f) {
            // During inference or no dropout, return gradients unchanged
            return gradOutput;
        }
        
        if (dropoutMask == null) {
            throw new IllegalStateException("Must call forward before backward");
        }
        
        int[] gradShape = gradOutput.getShape();
        FloatTensor gradOutputTensor = (FloatTensor) gradOutput;
        FloatTensor gradInput = TensorPool.get(gradShape);
        
        int totalElements = gradOutput.size();
        float[] gradOutData = gradOutputTensor.getPrimitiveData();
        float[] gradInData = gradInput.getPrimitiveData();
        float[] maskData = dropoutMask.getPrimitiveData();
        
        // Use parallel execution for large tensors
        if (totalElements > PARALLEL_THRESHOLD) {
            forkJoinPool.invoke(new DropoutBackwardTask(gradOutData, gradInData, maskData, 0, totalElements));
        } else {
            applyDropoutBackwardSequential(gradOutData, gradInData, maskData, totalElements);
        }
        
        gradInput.syncFromPrimitive();
        return gradInput;
    }
    
    private void applyDropoutBackwardSequential(float[] gradOut, float[] gradIn, float[] mask, int totalElements) {
        // Vectorized gradient masking - multiply by stored mask values
        for (int i = 0; i < totalElements; i++) {
            gradIn[i] = gradOut[i] * mask[i];
        }
    }
    
    /**
     * Parallel task for dropout forward pass
     */
    private static class DropoutForwardTask extends RecursiveAction {
        private static final int THRESHOLD = 1024;  // Work unit threshold
        
        private final float[] input;
        private final float[] output;
        private final float[] mask;
        private final float keepRate;
        private final float invKeepRate;
        private final int start;
        private final int end;
        
        DropoutForwardTask(float[] input, float[] output, float[] mask, 
                          float keepRate, float invKeepRate, int start, int end) {
            this.input = input;
            this.output = output;
            this.mask = mask;
            this.keepRate = keepRate;
            this.invKeepRate = invKeepRate;
            this.start = start;
            this.end = end;
        }
        
        @Override
        protected void compute() {
            if (end - start <= THRESHOLD) {
                computeDirectly();
            } else {
                int mid = (start + end) / 2;
                invokeAll(
                    new DropoutForwardTask(input, output, mask, keepRate, invKeepRate, start, mid),
                    new DropoutForwardTask(input, output, mask, keepRate, invKeepRate, mid, end)
                );
            }
        }
        
        private void computeDirectly() {
            // Use ThreadLocalRandom for better performance in parallel contexts
            Random random = ThreadLocalRandom.current();
            
            // Process elements in chunks for better cache locality
            int chunkSize = Math.min(64, end - start);
            
            for (int i = start; i < end; i += chunkSize) {
                int chunkEnd = Math.min(i + chunkSize, end);
                
                // Process chunk with unrolled loop for better performance
                for (int j = i; j < chunkEnd; j++) {
                    if (random.nextFloat() < keepRate) {
                        mask[j] = invKeepRate;
                        output[j] = input[j] * invKeepRate;
                    } else {
                        mask[j] = 0.0f;
                        output[j] = 0.0f;
                    }
                }
            }
        }
    }
    
    /**
     * Parallel task for dropout backward pass
     */
    private static class DropoutBackwardTask extends RecursiveAction {
        private static final int THRESHOLD = 1024;  // Work unit threshold
        
        private final float[] gradOut;
        private final float[] gradIn;
        private final float[] mask;
        private final int start;
        private final int end;
        
        DropoutBackwardTask(float[] gradOut, float[] gradIn, float[] mask, int start, int end) {
            this.gradOut = gradOut;
            this.gradIn = gradIn;
            this.mask = mask;
            this.start = start;
            this.end = end;
        }
        
        @Override
        protected void compute() {
            if (end - start <= THRESHOLD) {
                computeDirectly();
            } else {
                int mid = (start + end) / 2;
                invokeAll(
                    new DropoutBackwardTask(gradOut, gradIn, mask, start, mid),
                    new DropoutBackwardTask(gradOut, gradIn, mask, mid, end)
                );
            }
        }
        
        private void computeDirectly() {
            // Vectorized multiplication with loop unrolling for performance
            int i = start;
            int limit = end - 3;
            
            // Unrolled loop for better performance (4 elements at a time)
            for (; i < limit; i += 4) {
                gradIn[i] = gradOut[i] * mask[i];
                gradIn[i + 1] = gradOut[i + 1] * mask[i + 1];
                gradIn[i + 2] = gradOut[i + 2] * mask[i + 2];
                gradIn[i + 3] = gradOut[i + 3] * mask[i + 3];
            }
            
            // Handle remaining elements
            for (; i < end; i++) {
                gradIn[i] = gradOut[i] * mask[i];
            }
        }
    }
    
    /**
     * Alternative implementation using Bernoulli sampling for large-scale dropout
     * This is more memory efficient for very large tensors
     */
    public Tensor forwardMemoryEfficient(Tensor input) {
        if (!training || dropoutRate == 0.0f) {
            return input;
        }
        
        int[] inputShape = input.getShape();
        FloatTensor inputTensor = (FloatTensor) input;
        FloatTensor output = TensorPool.get(inputShape);
        
        int totalElements = input.size();
        float[] inputData = inputTensor.getPrimitiveData();
        float[] outputData = output.getPrimitiveData();
        
        // Generate dropout mask on-the-fly without storing it
        Random random = threadLocalRandom.get();
        
        if (totalElements > PARALLEL_THRESHOLD) {
            forkJoinPool.invoke(new MemoryEfficientDropoutTask(inputData, outputData, 
                                                             keepRate, invKeepRate, 0, totalElements));
        } else {
            for (int i = 0; i < totalElements; i++) {
                if (random.nextFloat() < keepRate) {
                    outputData[i] = inputData[i] * invKeepRate;
                } else {
                    outputData[i] = 0.0f;
                }
            }
        }
        
        output.syncFromPrimitive();
        return output;
    }
    
    /**
     * Memory-efficient parallel task that doesn't store dropout mask
     */
    private static class MemoryEfficientDropoutTask extends RecursiveAction {
        private static final int THRESHOLD = 1024;
        
        private final float[] input;
        private final float[] output;
        private final float keepRate;
        private final float invKeepRate;
        private final int start;
        private final int end;
        
        MemoryEfficientDropoutTask(float[] input, float[] output, float keepRate, float invKeepRate, int start, int end) {
            this.input = input;
            this.output = output;
            this.keepRate = keepRate;
            this.invKeepRate = invKeepRate;
            this.start = start;
            this.end = end;
        }
        
        @Override
        protected void compute() {
            if (end - start <= THRESHOLD) {
                Random random = ThreadLocalRandom.current();
                for (int i = start; i < end; i++) {
                    if (random.nextFloat() < keepRate) {
                        output[i] = input[i] * invKeepRate;
                    } else {
                        output[i] = 0.0f;
                    }
                }
            } else {
                int mid = (start + end) / 2;
                invokeAll(
                    new MemoryEfficientDropoutTask(input, output, keepRate, invKeepRate, start, mid),
                    new MemoryEfficientDropoutTask(input, output, keepRate, invKeepRate, mid, end)
                );
            }
        }
    }
    
    /**
     * Set dropout seed for reproducible results (useful for testing)
     */
    public static void setSeed(long seed) {
        threadLocalRandom.set(new Random(seed));
    }
    
    /**
     * Get effective dropout rate (accounts for training mode)
     */
    public float getEffectiveDropoutRate() {
        return training ? dropoutRate : 0.0f;
    }
    
    public float getDropoutRate() {
        return dropoutRate;
    }
    
    public float getKeepRate() {
        return keepRate;
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
        if (dropoutMask != null) {
            TensorPool.release(dropoutMask);
            dropoutMask = null;
        }
    }
    
    /**
     * Estimate memory usage for this layer
     */
    public long estimateMemoryUsage(int[] inputShape) {
        long elements = 1;
        for (int dim : inputShape) {
            elements *= dim;
        }
        // Each element uses 4 bytes (float) for mask during training
        return training ? elements * 4 : 0;
    }
    
    @Override
    public String toString() {
        return String.format("DropoutLayer(rate=%.3f, training=%s)", dropoutRate, training);
    }
}