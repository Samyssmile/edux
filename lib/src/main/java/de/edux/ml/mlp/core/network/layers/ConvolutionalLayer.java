package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class ConvolutionalLayer implements Layer {
    private final int numFilters;
    private final int filterSize;
    private final int inputHeight;
    private final int inputWidth;
    private final int inputChannels;

    //Flattened filters for better cache locality
    private final double[] filters; // [numFilters * inputChannels * filterSize * filterSize]
    private final double[] biases; // One bias per filter

    private Matrix lastInput;

    private transient final ForkJoinPool pool = ForkJoinPool.commonPool();

    public ConvolutionalLayer(int numFilters, int filterSize, int inputHeight, int inputWidth, int inputChannels) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputChannels = inputChannels;

        // Initialize filters and biases
        filters = new double[numFilters * inputChannels * filterSize * filterSize];
        biases = new double[numFilters];

        Random random = new Random();
        double stdDev = 1.0 / Math.sqrt(filterSize * filterSize * inputChannels);

        for (int f = 0; f < numFilters; f++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filters[f * inputChannels * filterSize * filterSize + c * filterSize * filterSize + i * filterSize + j] = random.nextGaussian() * stdDev;
                    }
                }
            }
            biases[f] = 0;
        }
    }

    @Override
    public Matrix forwardLayerbased(Matrix input) {
        this.lastInput = input;
        int batchSize = input.getCols();
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;

        // Output matrix dimensions: (numFilters * outputHeight * outputWidth) x batchSize
        Matrix output = new Matrix(numFilters * outputHeight * outputWidth, batchSize);

        // Parallelize convolution computation over the batch
                IntStream.range(0, batchSize).parallel().forEach(b -> {
                    double[] inputColumn = input.getColumn(b); // Flattened image input
                    double[][][] image = convertTo3DImage(inputColumn); // Convert to 3D image format

                    int outputIndex = 0;
                    for (int f = 0; f < numFilters; f++) {
                        double bias = biases[f];
                        for (int i = 0; i < outputHeight; i++) {
                            for (int j = 0; j < outputWidth; j++) {
                                double sum = 0;
                                // Convolution
                                for (int c = 0; c < inputChannels; c++) {
                                    int filterBase = f * inputChannels * filterSize * filterSize + c * filterSize * filterSize;
                                    for (int fi = 0; fi < filterSize; fi++) {
                                        for (int fj = 0; fj < filterSize; fj++) {
                                            sum += filters[filterBase + fi * filterSize + fj] * image[c][i + fi][j + fj];
                                        }
                                    }
                                }
                                sum += bias;
                                output.set(outputIndex++, b, sum);
                            }
                        }
                    }
                });

        return output;
    }

    @Override
    public void updateWeightsAndBias() {
        // Placeholder for weight/bias update logic if needed
    }

    @Override
    public Matrix backwardLayerBased(Matrix error, float learningRate) {
        int batchSize = error.getCols();
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;

        // Initialize gradients
        double[] filtersGradients = new double[numFilters * inputChannels * filterSize * filterSize];
        double[] biasesGradients = new double[numFilters];
        Matrix inputGradients = new Matrix(inputHeight * inputWidth * inputChannels, batchSize);

        // Parallelize backward pass over the batch
        pool.submit(() ->
                IntStream.range(0, batchSize).parallel().forEach(b -> {
                    double[] inputColumn = lastInput.getColumn(b);
                    double[][][] image = convertTo3DImage(inputColumn); // Convert last input to 3D image

                    double[] errorColumn = error.getColumn(b);
                    double[][][] errorVolume = new double[numFilters][outputHeight][outputWidth];
                    int errorIndex = 0;

                    for (int f = 0; f < numFilters; f++) {
                        for (int i = 0; i < outputHeight; i++) {
                            for (int j = 0; j < outputWidth; j++) {
                                errorVolume[f][i][j] = errorColumn[errorIndex++];
                            }
                        }
                    }

                    // Compute gradients
      /*              for (int f = 0; f < numFilters; f++) {*/
                    IntStream.range(0, numFilters).parallel().forEach(f -> {
                        for (int c = 0; c < inputChannels; c++) {
                            int filterBaseGrad = f * inputChannels * filterSize * filterSize + c * filterSize * filterSize;
                            for (int i = 0; i < filterSize; i++) {
                                for (int j = 0; j < filterSize; j++) {
                                    double sum = 0;
                                    for (int oi = 0; oi < outputHeight; oi++) {
                                        for (int oj = 0; oj < outputWidth; oj++) {
                                            sum += errorVolume[f][oi][oj] * image[c][oi + i][oj + j];
                                        }
                                    }
                                    filtersGradients[filterBaseGrad + i * filterSize + j] += sum / batchSize;
                                }
                            }
                        }
                        double biasSum = 0;
                        for (int oi = 0; oi < outputHeight; oi++) {
                            for (int oj = 0; oj < outputWidth; oj++) {
                                biasSum += errorVolume[f][oi][oj];
                            }
                        }
                        biasesGradients[f] += biasSum / batchSize;
                    });
                })
        ).join(); // Wait for gradients to be computed

        // Update filters and biases
/*        for (int f = 0; f < numFilters; f++) {*/
        IntStream.range(0, numFilters).parallel().forEach(f -> {
            for (int c = 0; c < inputChannels; c++) {
                int filterBaseGrad = f * inputChannels * filterSize * filterSize + c * filterSize * filterSize;
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filters[filterBaseGrad + i * filterSize + j] -= learningRate * filtersGradients[filterBaseGrad + i * filterSize + j];
                    }
                }
            }
            biases[f] -= learningRate * biasesGradients[f];
        });

        return inputGradients;
    }

    private double[][][] convertTo3DImage(double[] inputColumn) {
        double[][][] image = new double[inputChannels][inputHeight][inputWidth];
        for (int c = 0; c < inputChannels; c++) {
            int channelBase = c * inputHeight * inputWidth;
            for (int i = 0; i < inputHeight; i++) {
                for (int j = 0; j < inputWidth; j++) {
                    image[c][i][j] = inputColumn[channelBase + i * inputWidth + j];
                }
            }
        }
        return image;
    }

    @Override
    public String toString() {
        return "Convolutional Layer: " +
                "numFilters=" + numFilters +
                ", filterSize=" + filterSize +
                ", inputHeight=" + inputHeight +
                ", inputWidth=" + inputWidth +
                ", inputChannels=" + inputChannels;
    }
}