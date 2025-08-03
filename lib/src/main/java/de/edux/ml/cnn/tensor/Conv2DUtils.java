package de.edux.ml.cnn.tensor;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class Conv2DUtils {
    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();
    
    public static FloatTensor conv2d(FloatTensor input, FloatTensor weight, FloatTensor bias,
                                   int stride, int padding) {
        int[] inputShape = input.getShape();
        int[] weightShape = weight.getShape();
        
        if (inputShape.length != 4 || weightShape.length != 4) {
            throw new IllegalArgumentException("Input and weight must be 4D tensors");
        }
        
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        
        int outChannels = weightShape[0];
        int kernelHeight = weightShape[2];
        int kernelWidth = weightShape[3];
        
        int outHeight = (inHeight + 2 * padding - kernelHeight) / stride + 1;
        int outWidth = (inWidth + 2 * padding - kernelWidth) / stride + 1;
        
        FloatTensor output = TensorPool.get(new int[]{batch, outChannels, outHeight, outWidth});
        
        forkJoinPool.invoke(new Conv2DTask(
            input.getPrimitiveData(), weight.getPrimitiveData(), 
            bias != null ? bias.getPrimitiveData() : null,
            output.getPrimitiveData(),
            batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth,
            kernelHeight, kernelWidth, stride, padding,
            0, batch
        ));
        
        output.syncFromPrimitive();
        return output;
    }
    
    public static FloatTensor im2col(FloatTensor input, int kernelHeight, int kernelWidth,
                                   int stride, int padding, int outHeight, int outWidth) {
        int[] inputShape = input.getShape();
        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        
        int colHeight = channels * kernelHeight * kernelWidth;
        int colWidth = outHeight * outWidth;
        
        FloatTensor col = TensorPool.get(new int[]{batch, colHeight, colWidth});
        float[] inputData = input.getPrimitiveData();
        float[] colData = col.getPrimitiveData();
        
        for (int b = 0; b < batch; b++) {
            im2colSingle(inputData, colData, 
                        b * channels * height * width,
                        b * colHeight * colWidth,
                        channels, height, width,
                        kernelHeight, kernelWidth,
                        stride, padding, outHeight, outWidth);
        }
        
        col.syncFromPrimitive();
        return col;
    }
    
    private static void im2colSingle(float[] input, float[] col, int inputOffset, int colOffset,
                                   int channels, int height, int width,
                                   int kernelHeight, int kernelWidth,
                                   int stride, int padding, int outHeight, int outWidth) {
        int channelSize = height * width;
        
        for (int c = 0; c < channels; c++) {
            for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                    int colRow = c * kernelHeight * kernelWidth + kh * kernelWidth + kw;
                    
                    for (int oh = 0; oh < outHeight; oh++) {
                        for (int ow = 0; ow < outWidth; ow++) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            
                            int colIdx = colOffset + colRow * outHeight * outWidth + oh * outWidth + ow;
                            
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int inputIdx = inputOffset + c * channelSize + ih * width + iw;
                                col[colIdx] = input[inputIdx];
                            } else {
                                col[colIdx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
    
    private static class Conv2DTask extends RecursiveAction {
        private static final int THRESHOLD = 4;
        
        private final float[] input, weight, bias, output;
        private final int batch, inChannels, inHeight, inWidth;
        private final int outChannels, outHeight, outWidth;
        private final int kernelHeight, kernelWidth, stride, padding;
        private final int startBatch, endBatch;
        
        Conv2DTask(float[] input, float[] weight, float[] bias, float[] output,
                  int batch, int inChannels, int inHeight, int inWidth,
                  int outChannels, int outHeight, int outWidth,
                  int kernelHeight, int kernelWidth, int stride, int padding,
                  int startBatch, int endBatch) {
            this.input = input;
            this.weight = weight;
            this.bias = bias;
            this.output = output;
            this.batch = batch;
            this.inChannels = inChannels;
            this.inHeight = inHeight;
            this.inWidth = inWidth;
            this.outChannels = outChannels;
            this.outHeight = outHeight;
            this.outWidth = outWidth;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
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
                    new Conv2DTask(input, weight, bias, output,
                                 batch, inChannels, inHeight, inWidth,
                                 outChannels, outHeight, outWidth,
                                 kernelHeight, kernelWidth, stride, padding,
                                 startBatch, mid),
                    new Conv2DTask(input, weight, bias, output,
                                 batch, inChannels, inHeight, inWidth,
                                 outChannels, outHeight, outWidth,
                                 kernelHeight, kernelWidth, stride, padding,
                                 mid, endBatch)
                );
            }
        }
        
        private void computeDirectly() {
            int inputBatchStride = inChannels * inHeight * inWidth;
            int outputBatchStride = outChannels * outHeight * outWidth;
            int weightChannelStride = kernelHeight * kernelWidth;
            int weightFilterStride = inChannels * weightChannelStride;
            
            for (int b = startBatch; b < endBatch; b++) {
                int inputBatchOffset = b * inputBatchStride;
                int outputBatchOffset = b * outputBatchStride;
                
                for (int oc = 0; oc < outChannels; oc++) {
                    int outputChannelOffset = outputBatchOffset + oc * outHeight * outWidth;
                    int weightFilterOffset = oc * weightFilterStride;
                    
                    for (int oh = 0; oh < outHeight; oh++) {
                        for (int ow = 0; ow < outWidth; ow++) {
                            float sum = 0.0f;
                            
                            for (int ic = 0; ic < inChannels; ic++) {
                                int weightChannelOffset = weightFilterOffset + ic * weightChannelStride;
                                int inputChannelOffset = inputBatchOffset + ic * inHeight * inWidth;
                                
                                for (int kh = 0; kh < kernelHeight; kh++) {
                                    for (int kw = 0; kw < kernelWidth; kw++) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        
                                        if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                                            int inputIdx = inputChannelOffset + ih * inWidth + iw;
                                            int weightIdx = weightChannelOffset + kh * kernelWidth + kw;
                                            sum += input[inputIdx] * weight[weightIdx];
                                        }
                                    }
                                }
                            }
                            
                            if (bias != null) {
                                sum += bias[oc];
                            }
                            
                            int outputIdx = outputChannelOffset + oh * outWidth + ow;
                            output[outputIdx] = sum;
                        }
                    }
                }
            }
        }
    }
}