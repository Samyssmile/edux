package de.edux.ml.cnn.layer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import java.io.Serializable;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.Arrays;

/**
 * High-performance BatchNormalization layer with parallel execution
 * Supports both 2D (FC) and 4D (Conv) inputs with optimized memory usage
 */
public class BatchNormalizationLayer implements Layer, Serializable {
    private final int numFeatures;
    private final float epsilon;
    private final float momentum;
    
    // Learnable parameters
    private FloatTensor gamma;  // Scale parameter
    private FloatTensor beta;   // Shift parameter
    
    // Running statistics for inference
    private FloatTensor runningMean;
    private FloatTensor runningVar;
    
    // Gradients
    private FloatTensor gammaGradients;
    private FloatTensor betaGradients;
    
    // Training state
    private boolean training = true;
    private Tensor lastInput;
    private FloatTensor lastNormalized;
    private FloatTensor batchMean;
    private FloatTensor batchVar;
    
    // Performance optimization
    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();
    private static final int PARALLEL_THRESHOLD = 1024;
    
    public BatchNormalizationLayer(int numFeatures) {
        this(numFeatures, 1e-5f, 0.1f);
    }
    
    public BatchNormalizationLayer(int numFeatures, float epsilon, float momentum) {
        this.numFeatures = numFeatures;
        this.epsilon = epsilon;
        this.momentum = momentum;
        
        initializeParameters();
    }
    
    private void initializeParameters() {
        // Initialize gamma to 1 (identity scaling)
        gamma = FloatTensor.ones(numFeatures);
        
        // Initialize beta to 0 (no shift)
        beta = FloatTensor.zeros(numFeatures);
        
        // Initialize running statistics
        runningMean = FloatTensor.zeros(numFeatures);
        runningVar = FloatTensor.ones(numFeatures);
    }
    
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        
        int[] inputShape = input.getShape();
        if (inputShape.length != 2 && inputShape.length != 4) {
            throw new IllegalArgumentException("BatchNorm supports 2D [batch, features] or 4D [batch, channels, height, width] input");
        }
        
        FloatTensor inputTensor = (FloatTensor) input;
        
        if (inputShape.length == 4) {
            return forwardConv(inputTensor, inputShape);
        } else {
            return forwardFC(inputTensor, inputShape);
        }
    }
    
    private FloatTensor forwardFC(FloatTensor input, int[] shape) {
        int batchSize = shape[0];
        int features = shape[1];
        
        if (features != numFeatures) {
            throw new IllegalArgumentException("Expected " + numFeatures + " features, got " + features);
        }
        
        FloatTensor output = TensorPool.get(shape);
        float[] inputData = input.getPrimitiveData();
        float[] outputData = output.getPrimitiveData();
        
        if (training) {
            // Compute batch statistics
            batchMean = TensorPool.get(new int[]{numFeatures});
            batchVar = TensorPool.get(new int[]{numFeatures});
            
            computeBatchStatisticsFC(inputData, batchSize, features);
            
            // Update running statistics
            updateRunningStatistics();
            
            // Normalize and apply affine transformation
            normalizeAndTransformFC(inputData, outputData, batchSize, features, 
                                  batchMean.getPrimitiveData(), batchVar.getPrimitiveData());
        } else {
            // Use running statistics for inference
            normalizeAndTransformFC(inputData, outputData, batchSize, features,
                                  runningMean.getPrimitiveData(), runningVar.getPrimitiveData());
        }
        
        output.syncFromPrimitive();
        return output;
    }
    
    private FloatTensor forwardConv(FloatTensor input, int[] shape) {
        int batchSize = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        
        if (channels != numFeatures) {
            throw new IllegalArgumentException("Expected " + numFeatures + " channels, got " + channels);
        }
        
        FloatTensor output = TensorPool.get(shape);
        float[] inputData = input.getPrimitiveData();
        float[] outputData = output.getPrimitiveData();
        
        if (training) {
            batchMean = TensorPool.get(new int[]{numFeatures});
            batchVar = TensorPool.get(new int[]{numFeatures});
            
            computeBatchStatisticsConv(inputData, batchSize, channels, height, width);
            updateRunningStatistics();
            
            normalizeAndTransformConv(inputData, outputData, batchSize, channels, height, width,
                                    batchMean.getPrimitiveData(), batchVar.getPrimitiveData());
        } else {
            normalizeAndTransformConv(inputData, outputData, batchSize, channels, height, width,
                                    runningMean.getPrimitiveData(), runningVar.getPrimitiveData());
        }
        
        output.syncFromPrimitive();
        
        // Cache normalized values for backward pass
        if (training) {
            if (lastNormalized != null) {
                TensorPool.release(lastNormalized);
            }
            lastNormalized = TensorPool.get(shape);
            computeNormalizedValues(inputData, lastNormalized.getPrimitiveData(), 
                                  batchSize, channels, height, width, 
                                  batchMean.getPrimitiveData(), batchVar.getPrimitiveData());
            lastNormalized.syncFromPrimitive();
        }
        
        return output;
    }
    
    private void computeBatchStatisticsFC(float[] input, int batchSize, int features) {
        float[] meanData = batchMean.getPrimitiveData();
        float[] varData = batchVar.getPrimitiveData();
        
        // Parallel computation of mean
        if (batchSize * features > PARALLEL_THRESHOLD) {
            forkJoinPool.invoke(new ComputeMeanTaskFC(input, meanData, batchSize, features, 0, features));
        } else {
            computeMeanFC(input, meanData, batchSize, features);
        }
        
        // Parallel computation of variance
        if (batchSize * features > PARALLEL_THRESHOLD) {
            forkJoinPool.invoke(new ComputeVarianceTaskFC(input, meanData, varData, batchSize, features, 0, features));
        } else {
            computeVarianceFC(input, meanData, varData, batchSize, features);
        }
        
        batchMean.syncFromPrimitive();
        batchVar.syncFromPrimitive();
    }
    
    private void computeBatchStatisticsConv(float[] input, int batchSize, int channels, int height, int width) {
        float[] meanData = batchMean.getPrimitiveData();
        float[] varData = batchVar.getPrimitiveData();
        
        int spatialSize = height * width;
        int totalSamples = batchSize * spatialSize;
        
        // Parallel computation of mean and variance for conv layers
        if (channels > 8) {
            forkJoinPool.invoke(new ComputeStatsTaskConv(input, meanData, varData, batchSize, channels, 
                                                       height, width, totalSamples, 0, channels, true));
        } else {
            computeStatsConv(input, meanData, varData, batchSize, channels, height, width, totalSamples);
        }
        
        batchMean.syncFromPrimitive();
        batchVar.syncFromPrimitive();
    }
    
    private void computeMeanFC(float[] input, float[] mean, int batchSize, int features) {
        Arrays.fill(mean, 0.0f);
        
        for (int f = 0; f < features; f++) {
            for (int b = 0; b < batchSize; b++) {
                mean[f] += input[b * features + f];
            }
            mean[f] /= batchSize;
        }
    }
    
    private void computeVarianceFC(float[] input, float[] mean, float[] variance, int batchSize, int features) {
        Arrays.fill(variance, 0.0f);
        
        for (int f = 0; f < features; f++) {
            for (int b = 0; b < batchSize; b++) {
                float diff = input[b * features + f] - mean[f];
                variance[f] += diff * diff;
            }
            variance[f] /= batchSize;
        }
    }
    
    private void computeStatsConv(float[] input, float[] mean, float[] variance, 
                                int batchSize, int channels, int height, int width, int totalSamples) {
        Arrays.fill(mean, 0.0f);
        Arrays.fill(variance, 0.0f);
        
        int spatialSize = height * width;
        
        // Compute mean
        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batchSize; b++) {
                int channelOffset = b * channels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    mean[c] += input[channelOffset + s];
                }
            }
            mean[c] /= totalSamples;
        }
        
        // Compute variance
        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batchSize; b++) {
                int channelOffset = b * channels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    float diff = input[channelOffset + s] - mean[c];
                    variance[c] += diff * diff;
                }
            }
            variance[c] /= totalSamples;
        }
    }
    
    private void normalizeAndTransformFC(float[] input, float[] output, int batchSize, int features,
                                       float[] mean, float[] variance) {
        float[] gammaData = gamma.getPrimitiveData();
        float[] betaData = beta.getPrimitiveData();
        
        for (int b = 0; b < batchSize; b++) {
            for (int f = 0; f < features; f++) {
                int idx = b * features + f;
                float normalized = (input[idx] - mean[f]) / (float) Math.sqrt(variance[f] + epsilon);
                output[idx] = gammaData[f] * normalized + betaData[f];
            }
        }
    }
    
    private void normalizeAndTransformConv(float[] input, float[] output, int batchSize, int channels, 
                                         int height, int width, float[] mean, float[] variance) {
        float[] gammaData = gamma.getPrimitiveData();
        float[] betaData = beta.getPrimitiveData();
        
        int spatialSize = height * width;
        
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                int channelOffset = b * channels * spatialSize + c * spatialSize;
                float invStd = 1.0f / (float) Math.sqrt(variance[c] + epsilon);
                float scaledGamma = gammaData[c] * invStd;
                float shiftedBeta = betaData[c] - mean[c] * scaledGamma;
                
                for (int s = 0; s < spatialSize; s++) {
                    int idx = channelOffset + s;
                    output[idx] = input[idx] * scaledGamma + shiftedBeta;
                }
            }
        }
    }
    
    private void computeNormalizedValues(float[] input, float[] normalized, int batchSize, int channels,
                                       int height, int width, float[] mean, float[] variance) {
        int spatialSize = height * width;
        
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                int channelOffset = b * channels * spatialSize + c * spatialSize;
                float invStd = 1.0f / (float) Math.sqrt(variance[c] + epsilon);
                
                for (int s = 0; s < spatialSize; s++) {
                    int idx = channelOffset + s;
                    normalized[idx] = (input[idx] - mean[c]) * invStd;
                }
            }
        }
    }
    
    private void updateRunningStatistics() {
        float[] runningMeanData = runningMean.getPrimitiveData();
        float[] runningVarData = runningVar.getPrimitiveData();
        float[] batchMeanData = batchMean.getPrimitiveData();
        float[] batchVarData = batchVar.getPrimitiveData();
        
        for (int i = 0; i < numFeatures; i++) {
            runningMeanData[i] = (1.0f - momentum) * runningMeanData[i] + momentum * batchMeanData[i];
            runningVarData[i] = (1.0f - momentum) * runningVarData[i] + momentum * batchVarData[i];
        }
        
        runningMean.syncFromPrimitive();
        runningVar.syncFromPrimitive();
    }
    
    @Override
    public Tensor backward(Tensor gradOutput) {
        if (lastInput == null) {
            throw new IllegalStateException("Must call forward before backward");
        }
        
        int[] inputShape = lastInput.getShape();
        FloatTensor gradInput = TensorPool.get(inputShape);
        
        if (inputShape.length == 4) {
            backwardConv((FloatTensor) gradOutput, gradInput, inputShape);
        } else {
            backwardFC((FloatTensor) gradOutput, gradInput, inputShape);
        }
        
        return gradInput;
    }
    
    private void backwardFC(FloatTensor gradOutput, FloatTensor gradInput, int[] shape) {
        int batchSize = shape[0];
        int features = shape[1];
        
        float[] gradOutData = gradOutput.getPrimitiveData();
        float[] gradInData = gradInput.getPrimitiveData();
        float[] inputData = ((FloatTensor) lastInput).getPrimitiveData();
        float[] meanData = batchMean.getPrimitiveData();
        float[] varData = batchVar.getPrimitiveData();
        float[] gammaData = gamma.getPrimitiveData();
        
        // Initialize gradients
        if (gammaGradients == null) {
            gammaGradients = TensorPool.get(new int[]{numFeatures});
            betaGradients = TensorPool.get(new int[]{numFeatures});
        }
        
        float[] gammaGradData = gammaGradients.getPrimitiveData();
        float[] betaGradData = betaGradients.getPrimitiveData();
        Arrays.fill(gammaGradData, 0.0f);
        Arrays.fill(betaGradData, 0.0f);
        
        // Compute parameter gradients and input gradients
        for (int f = 0; f < features; f++) {
            float invStd = 1.0f / (float) Math.sqrt(varData[f] + epsilon);
            float sumGradOut = 0.0f;
            float sumGradOutNorm = 0.0f;
            
            // Accumulate gradients for this feature
            for (int b = 0; b < batchSize; b++) {
                int idx = b * features + f;
                float gradOut = gradOutData[idx];
                float normalized = (inputData[idx] - meanData[f]) * invStd;
                
                sumGradOut += gradOut;
                sumGradOutNorm += gradOut * normalized;
                
                gammaGradData[f] += gradOut * normalized;
                betaGradData[f] += gradOut;
            }
            
            // Compute input gradients
            float scale = invStd / batchSize;
            for (int b = 0; b < batchSize; b++) {
                int idx = b * features + f;
                float normalized = (inputData[idx] - meanData[f]) * invStd;
                gradInData[idx] = gammaData[f] * scale * (batchSize * gradOutData[idx] - sumGradOut - normalized * sumGradOutNorm);
            }
        }
        
        gammaGradients.syncFromPrimitive();
        betaGradients.syncFromPrimitive();
        gradInput.syncFromPrimitive();
    }
    
    private void backwardConv(FloatTensor gradOutput, FloatTensor gradInput, int[] shape) {
        int batchSize = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int spatialSize = height * width;
        
        float[] gradOutData = gradOutput.getPrimitiveData();
        float[] gradInData = gradInput.getPrimitiveData();
        float[] normalizedData = lastNormalized.getPrimitiveData();
        float[] gammaData = gamma.getPrimitiveData();
        float[] varData = batchVar.getPrimitiveData();
        
        // Initialize gradients
        if (gammaGradients == null) {
            gammaGradients = TensorPool.get(new int[]{numFeatures});
            betaGradients = TensorPool.get(new int[]{numFeatures});
        }
        
        float[] gammaGradData = gammaGradients.getPrimitiveData();
        float[] betaGradData = betaGradients.getPrimitiveData();
        Arrays.fill(gammaGradData, 0.0f);
        Arrays.fill(betaGradData, 0.0f);
        
        int totalSpatial = batchSize * spatialSize;
        
        // Parallel backward computation for conv layers
        if (channels > 8) {
            forkJoinPool.invoke(new BackwardTaskConv(gradOutData, gradInData, normalizedData, 
                                                   gammaData, varData, gammaGradData, betaGradData,
                                                   batchSize, channels, spatialSize, totalSpatial,
                                                   0, channels));
        } else {
            backwardConvDirect(gradOutData, gradInData, normalizedData, gammaData, varData,
                             gammaGradData, betaGradData, batchSize, channels, spatialSize, totalSpatial);
        }
        
        gammaGradients.syncFromPrimitive();
        betaGradients.syncFromPrimitive();
        gradInput.syncFromPrimitive();
    }
    
    private void backwardConvDirect(float[] gradOut, float[] gradIn, float[] normalized, 
                                  float[] gamma, float[] variance, float[] gammaGrad, float[] betaGrad,
                                  int batchSize, int channels, int spatialSize, int totalSpatial) {
        for (int c = 0; c < channels; c++) {
            float invStd = 1.0f / (float) Math.sqrt(variance[c] + epsilon);
            float sumGradOut = 0.0f;
            float sumGradOutNorm = 0.0f;
            
            // Accumulate gradients for this channel
            for (int b = 0; b < batchSize; b++) {
                int channelOffset = b * channels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    int idx = channelOffset + s;
                    float gradOutVal = gradOut[idx];
                    float normVal = normalized[idx];
                    
                    sumGradOut += gradOutVal;
                    sumGradOutNorm += gradOutVal * normVal;
                    
                    gammaGrad[c] += gradOutVal * normVal;
                    betaGrad[c] += gradOutVal;
                }
            }
            
            // Compute input gradients
            float scale = gamma[c] * invStd / totalSpatial;
            for (int b = 0; b < batchSize; b++) {
                int channelOffset = b * channels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++) {
                    int idx = channelOffset + s;
                    float normVal = normalized[idx];
                    gradIn[idx] = scale * (totalSpatial * gradOut[idx] - sumGradOut - normVal * sumGradOutNorm);
                }
            }
        }
    }
    
    // Parallel tasks for performance optimization
    private static class ComputeMeanTaskFC extends RecursiveAction {
        private static final int THRESHOLD = 64;
        private final float[] input;
        private final float[] mean;
        private final int batchSize;
        private final int features;
        private final int startFeature;
        private final int endFeature;
        
        ComputeMeanTaskFC(float[] input, float[] mean, int batchSize, int features, int startFeature, int endFeature) {
            this.input = input;
            this.mean = mean;
            this.batchSize = batchSize;
            this.features = features;
            this.startFeature = startFeature;
            this.endFeature = endFeature;
        }
        
        @Override
        protected void compute() {
            if (endFeature - startFeature <= THRESHOLD) {
                for (int f = startFeature; f < endFeature; f++) {
                    float sum = 0.0f;
                    for (int b = 0; b < batchSize; b++) {
                        sum += input[b * features + f];
                    }
                    mean[f] = sum / batchSize;
                }
            } else {
                int mid = (startFeature + endFeature) / 2;
                invokeAll(
                    new ComputeMeanTaskFC(input, mean, batchSize, features, startFeature, mid),
                    new ComputeMeanTaskFC(input, mean, batchSize, features, mid, endFeature)
                );
            }
        }
    }
    
    private static class ComputeVarianceTaskFC extends RecursiveAction {
        private static final int THRESHOLD = 64;
        private final float[] input;
        private final float[] mean;
        private final float[] variance;
        private final int batchSize;
        private final int features;
        private final int startFeature;
        private final int endFeature;
        
        ComputeVarianceTaskFC(float[] input, float[] mean, float[] variance, int batchSize, int features, int startFeature, int endFeature) {
            this.input = input;
            this.mean = mean;
            this.variance = variance;
            this.batchSize = batchSize;
            this.features = features;
            this.startFeature = startFeature;
            this.endFeature = endFeature;
        }
        
        @Override
        protected void compute() {
            if (endFeature - startFeature <= THRESHOLD) {
                for (int f = startFeature; f < endFeature; f++) {
                    float sumSq = 0.0f;
                    for (int b = 0; b < batchSize; b++) {
                        float diff = input[b * features + f] - mean[f];
                        sumSq += diff * diff;
                    }
                    variance[f] = sumSq / batchSize;
                }
            } else {
                int mid = (startFeature + endFeature) / 2;
                invokeAll(
                    new ComputeVarianceTaskFC(input, mean, variance, batchSize, features, startFeature, mid),
                    new ComputeVarianceTaskFC(input, mean, variance, batchSize, features, mid, endFeature)
                );
            }
        }
    }
    
    private static class ComputeStatsTaskConv extends RecursiveAction {
        private static final int THRESHOLD = 8;
        private final float[] input;
        private final float[] mean;
        private final float[] variance;
        private final int batchSize;
        private final int channels;
        private final int height;
        private final int width;
        private final int totalSamples;
        private final int startChannel;
        private final int endChannel;
        private final boolean computeMean;
        
        ComputeStatsTaskConv(float[] input, float[] mean, float[] variance, int batchSize, int channels,
                           int height, int width, int totalSamples, int startChannel, int endChannel, boolean computeMean) {
            this.input = input;
            this.mean = mean;
            this.variance = variance;
            this.batchSize = batchSize;
            this.channels = channels;
            this.height = height;
            this.width = width;
            this.totalSamples = totalSamples;
            this.startChannel = startChannel;
            this.endChannel = endChannel;
            this.computeMean = computeMean;
        }
        
        @Override
        protected void compute() {
            if (endChannel - startChannel <= THRESHOLD) {
                computeStatsDirect();
            } else {
                int mid = (startChannel + endChannel) / 2;
                invokeAll(
                    new ComputeStatsTaskConv(input, mean, variance, batchSize, channels, height, width,
                                           totalSamples, startChannel, mid, computeMean),
                    new ComputeStatsTaskConv(input, mean, variance, batchSize, channels, height, width,
                                           totalSamples, mid, endChannel, computeMean)
                );
            }
        }
        
        private void computeStatsDirect() {
            int spatialSize = height * width;
            
            if (computeMean) {
                // Compute mean
                for (int c = startChannel; c < endChannel; c++) {
                    float sum = 0.0f;
                    for (int b = 0; b < batchSize; b++) {
                        int channelOffset = b * channels * spatialSize + c * spatialSize;
                        for (int s = 0; s < spatialSize; s++) {
                            sum += input[channelOffset + s];
                        }
                    }
                    mean[c] = sum / totalSamples;
                }
            }
            
            // Compute variance
            for (int c = startChannel; c < endChannel; c++) {
                float sumSq = 0.0f;
                for (int b = 0; b < batchSize; b++) {
                    int channelOffset = b * channels * spatialSize + c * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        float diff = input[channelOffset + s] - mean[c];
                        sumSq += diff * diff;
                    }
                }
                variance[c] = sumSq / totalSamples;
            }
        }
    }
    
    private static class BackwardTaskConv extends RecursiveAction {
        private static final int THRESHOLD = 4;
        private final float[] gradOut;
        private final float[] gradIn;
        private final float[] normalized;
        private final float[] gamma;
        private final float[] variance;
        private final float[] gammaGrad;
        private final float[] betaGrad;
        private final int batchSize;
        private final int channels;
        private final int spatialSize;
        private final int totalSpatial;
        private final int startChannel;
        private final int endChannel;
        
        BackwardTaskConv(float[] gradOut, float[] gradIn, float[] normalized, float[] gamma, float[] variance,
                        float[] gammaGrad, float[] betaGrad, int batchSize, int channels, int spatialSize,
                        int totalSpatial, int startChannel, int endChannel) {
            this.gradOut = gradOut;
            this.gradIn = gradIn;
            this.normalized = normalized;
            this.gamma = gamma;
            this.variance = variance;
            this.gammaGrad = gammaGrad;
            this.betaGrad = betaGrad;
            this.batchSize = batchSize;
            this.channels = channels;
            this.spatialSize = spatialSize;
            this.totalSpatial = totalSpatial;
            this.startChannel = startChannel;
            this.endChannel = endChannel;
        }
        
        @Override
        protected void compute() {
            if (endChannel - startChannel <= THRESHOLD) {
                computeBackwardDirect();
            } else {
                int mid = (startChannel + endChannel) / 2;
                invokeAll(
                    new BackwardTaskConv(gradOut, gradIn, normalized, gamma, variance, gammaGrad, betaGrad,
                                       batchSize, channels, spatialSize, totalSpatial, startChannel, mid),
                    new BackwardTaskConv(gradOut, gradIn, normalized, gamma, variance, gammaGrad, betaGrad,
                                       batchSize, channels, spatialSize, totalSpatial, mid, endChannel)
                );
            }
        }
        
        private void computeBackwardDirect() {
            final float epsilon = 1e-5f;
            
            for (int c = startChannel; c < endChannel; c++) {
                float invStd = 1.0f / (float) Math.sqrt(variance[c] + epsilon);
                float sumGradOut = 0.0f;
                float sumGradOutNorm = 0.0f;
                
                // Accumulate gradients for this channel
                for (int b = 0; b < batchSize; b++) {
                    int channelOffset = b * channels * spatialSize + c * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        int idx = channelOffset + s;
                        float gradOutVal = gradOut[idx];
                        float normVal = normalized[idx];
                        
                        sumGradOut += gradOutVal;
                        sumGradOutNorm += gradOutVal * normVal;
                        
                        synchronized (gammaGrad) {
                            gammaGrad[c] += gradOutVal * normVal;
                        }
                        synchronized (betaGrad) {
                            betaGrad[c] += gradOutVal;
                        }
                    }
                }
                
                // Compute input gradients
                float scale = gamma[c] * invStd / totalSpatial;
                for (int b = 0; b < batchSize; b++) {
                    int channelOffset = b * channels * spatialSize + c * spatialSize;
                    for (int s = 0; s < spatialSize; s++) {
                        int idx = channelOffset + s;
                        float normVal = normalized[idx];
                        gradIn[idx] = scale * (totalSpatial * gradOut[idx] - sumGradOut - normVal * sumGradOutNorm);
                    }
                }
            }
        }
    }
    
    // Getters for parameters and gradients
    public FloatTensor getGamma() { return gamma; }
    public FloatTensor getBeta() { return beta; }
    public FloatTensor getGammaGradients() { return gammaGradients; }
    public FloatTensor getBetaGradients() { return betaGradients; }
    public FloatTensor getRunningMean() { return runningMean; }
    public FloatTensor getRunningVar() { return runningVar; }
    
    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    @Override
    public boolean isTraining() {
        return training;
    }
    
    public void zeroGradients() {
        if (gammaGradients != null) {
            TensorPool.release(gammaGradients);
            gammaGradients = null;
        }
        if (betaGradients != null) {
            TensorPool.release(betaGradients);
            betaGradients = null;
        }
    }
    
    @Override
    public void cleanup() {
        if (lastNormalized != null) {
            TensorPool.release(lastNormalized);
            lastNormalized = null;
        }
        if (batchMean != null) {
            TensorPool.release(batchMean);
            batchMean = null;
        }
        if (batchVar != null) {
            TensorPool.release(batchVar);
            batchVar = null;
        }
        zeroGradients();
    }
}