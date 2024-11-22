package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;

import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class FlattenLayer implements Layer {
    private int numChannels;
    private int inputHeight;
    private int inputWidth;
    private int outputSize;
    private Matrix lastInput;

    private transient final ForkJoinPool pool = ForkJoinPool.commonPool();


    public FlattenLayer(int numChannels, int inputHeight, int inputWidth) {
        this.numChannels = numChannels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.outputSize = numChannels * inputHeight * inputWidth; // Outputgröße berechnen
    }

    @Override
    public Matrix forwardLayerbased(Matrix input) {
        this.lastInput = input; // Speichere das Eingabematrix für die Rückwärtsweiterleitung
        int batchSize = input.getCols(); // Batch-Größe

        // Erstelle die Ausgabematrix mit der richtigen Größe
        Matrix output = new Matrix(outputSize, batchSize);
        pool.submit(() ->
                IntStream.range(0, batchSize).parallel().forEach(b -> {
                    double[] inputColumn = input.getColumn(b); // Hole die Spalte (ein "Bild" im Batch)
                    double[] flattened = new double[outputSize]; // Dieser Vektor wird ein Bild plattmachen
                    int index = 0;

                    // Durchlaufe jedes Element in (Kanäle x Höhe x Breite) und flach es.
                    for (int c = 0; c < numChannels; c++) {
                        for (int i = 0; i < inputHeight; i++) {
                            for (int j = 0; j < inputWidth; j++) {
                                flattened[index++] = inputColumn[c * inputHeight * inputWidth + i * inputWidth + j];
                            }
                        }
                    }

                    // Speichere den flachgemachten Vektor in der Output-Matrix.
                    output.setColumn(b, flattened);
                })).join();

        return output;
    }

    @Override
    public Matrix backwardLayerBased(Matrix error, float learningRate) {
        // Rückwärtsweiterleitung ist einfacher: Der Fehler wird einfach in die ursprüngliche Form gebracht
        // Dies geschieht, indem man das Format [NumChannels * InputHeight * InputWidth] in [NumChannels, InputHeight, InputWidth] zurück umwandelt.
        int batchSize = error.getCols();
        Matrix errorReshaped = new Matrix(numChannels * inputHeight * inputWidth, batchSize);

        // Prinzip bleibt dasselbe: Wir transformieren den Fehler zurück in die Form der vorherigen Schicht.
        for (int b = 0; b < batchSize; b++) {
            double[] errorColumn = error.getColumn(b);
            double[] reshaped = new double[inputHeight * inputWidth * numChannels];

            // Deflatiere die Zeilen (reverse operation)
            for (int i = 0; i < reshaped.length; i++) {
                reshaped[i] = errorColumn[i];
            }
            errorReshaped.setColumn(b, reshaped);
        }

/*        pool.submit(() ->
                IntStream.range(0, batchSize).parallel().forEach(b -> {
                    double[] errorColumn = error.getColumn(b);
                    double[] reshaped = new double[inputHeight * inputWidth * numChannels];

                    // Deflatiere die Zeilen (reverse operation)
                    for (int i = 0; i < reshaped.length; i++) {
                        reshaped[i] = errorColumn[i];
                    }
                    errorReshaped.setColumn(b, reshaped);
                })).join();*/



        return errorReshaped;
    }

    @Override
    public void updateWeightsAndBias() {
        // Keine Gewichte oder Biases, da der FlattenLayer nur eine Transformation ist.
    }

    @Override
    public String toString() {
        return "FlattenLayer: outputSize=" + outputSize;
    }
}