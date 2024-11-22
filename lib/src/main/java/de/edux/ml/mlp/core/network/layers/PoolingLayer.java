package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class PoolingLayer implements Layer {

    private int poolHeight;
    private int poolWidth;
    private int stride;
    private int inputHeight;
    private int inputWidth;
    private int numChannels;
    private int outputHeight;
    private int outputWidth;

    private transient final ForkJoinPool pool = ForkJoinPool.commonPool();


    // Speichert die Positionen der Maxima für die Backpropagation
    private Map<Integer, int[][][]> maxIndices;

    private Matrix lastInput;

    public PoolingLayer(int numChannels, int inputHeight, int inputWidth, int poolHeight, int poolWidth, int stride) {
        this.numChannels = numChannels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
        this.stride = stride;

        // Berechnet die Ausgabegröße
        this.outputHeight = (inputHeight - poolHeight) / stride + 1;
        this.outputWidth = (inputWidth - poolWidth) / stride + 1;

        this.maxIndices = new HashMap<>();
    }

    @Override
    public Matrix forwardLayerbased(Matrix input) {
        this.lastInput = input;
        int batchSize = input.getCols();

        // Ausgabematrix (numChannels * outputHeight * outputWidth) x batchSize
        Matrix output = new Matrix(numChannels * outputHeight * outputWidth, batchSize);

        pool.submit(() -> IntStream.range(0, batchSize).parallel().forEach(b -> {
            double[] inputColumn = input.getColumn(b);
            double[][][] inputTensor = convertTo3D(inputColumn);

            int[][][] sampleMaxIndices = new int[numChannels][outputHeight][outputWidth];
            int outputIndex = 0;

            for (int c = 0; c < numChannels; c++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxI = -1, maxJ = -1;

                        for (int m = 0; m < poolHeight; m++) {
                            for (int n = 0; n < poolWidth; n++) {
                                int currI = i * stride + m;
                                int currJ = j * stride + n;
                                if (currI < inputHeight && currJ < inputWidth) {
                                    double val = inputTensor[c][currI][currJ];
                                    if (val > maxVal) {
                                        maxVal = val;
                                        maxI = currI;
                                        maxJ = currJ;
                                    }
                                }
                            }
                        }

                        // Setze das Maximum im Output
                        output.set(outputIndex++, b, maxVal);

                        // Speichere den Index des Maximums für die Backpropagation
                        sampleMaxIndices[c][i][j] = maxI * inputWidth + maxJ;  // Flacher Index
                    }
                }
            }

            // Indizes für die Backpropagation speichern
            maxIndices.put(b, sampleMaxIndices);
        })).join();

        return output;
    }

    @Override
    public Matrix backwardLayerBased(Matrix error, float learningRate) {
        int batchSize = error.getCols();
        Matrix inputGradients = new Matrix(numChannels * inputHeight * inputWidth, batchSize);


        pool.submit(() -> IntStream.range(0, batchSize).forEach(b -> {
            int[][][] sampleMaxIndices = maxIndices.get(b);
            double[] errorColumn = error.getColumn(b);
            double[][][] inputGrad = new double[numChannels][inputHeight][inputWidth];

            // Fehler auf die Indizes der Maxima verteilen
            int errorIndex = 0;
            for (int c = 0; c < numChannels; c++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        int maxIndex = sampleMaxIndices[c][i][j];
                        int maxI = maxIndex / inputWidth;
                        int maxJ = maxIndex % inputWidth;
try{
                        inputGrad[c][maxI][maxJ] += errorColumn[errorIndex++];
}catch(Exception e){
    System.out.println("errorIndex: "+errorIndex);
}
                    }
                }
            }

            // Gradienten flach machen und in inputGradients speichern
            double[] inputGradColumn = new double[numChannels * inputHeight * inputWidth];
            int gradIndex = 0;
            for (int c = 0; c < numChannels; c++) {
                for (int i = 0; i < inputHeight; i++) {
                    for (int j = 0; j < inputWidth; j++) {
                        inputGradColumn[gradIndex++] = inputGrad[c][i][j];
                    }
                }
            }

            inputGradients.setColumn(b, inputGradColumn);
        })).join();
        return inputGradients;
    }

    @Override
    public void updateWeightsAndBias() {
        // Keine Gewichte oder Biases, die aktualisiert werden müssen.
    }

    @Override
    public String toString() {
        return "PoolingLayer: poolSize=" + poolHeight + "x" + poolWidth + ", stride=" + stride;
    }

    // Konvertiert ein flaches Array zu einem 3D Tensor
    private double[][][] convertTo3D(double[] inputColumn) {
        double[][][] inputTensor = new double[numChannels][inputHeight][inputWidth];
        int index = 0;
        for (int c = 0; c < numChannels; c++) {
            for (int i = 0; i < inputHeight; i++) {
                for (int j = 0; j < inputWidth; j++) {
                    inputTensor[c][i][j] = inputColumn[index++];
                }
            }
        }
        return inputTensor;
    }
}