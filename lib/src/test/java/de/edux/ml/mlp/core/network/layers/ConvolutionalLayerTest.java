package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.tensor.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


/**
 * @author Samuel Abramov
 */
class ConvolutionalLayerTest {

    @Test
    public void testSimpleConvolution() {
        // Test-Parameter
        int numFilters = 2;
        int filterSize = 2;
        int inputHeight = 4;
        int inputWidth = 4;
        int inputChannels = 1;

        // Layer erstellen
        ConvolutionalLayer layer = new ConvolutionalLayer(
                numFilters, filterSize, inputHeight, inputWidth, inputChannels
        );

        // Testdaten erstellen: Ein 4x4 Eingabebild
        Matrix input = new Matrix(inputHeight * inputWidth, 1);  // Eine Spalte für ein Bild
        // Einfaches Testmuster: aufsteigende Zahlen
        for (int i = 0; i < inputHeight * inputWidth; i++) {
            input.set(i, 0, i + 1);
        }

        // Forward Pass durchführen
        Matrix output = layer.forwardLayerbased(input);

        // Überprüfungen
        // 1. Überprüfe Output-Dimensionen
        int expectedOutputHeight = inputHeight - filterSize + 1;  // 3
        int expectedOutputWidth = inputWidth - filterSize + 1;    // 3
        int expectedOutputSize = numFilters * expectedOutputHeight * expectedOutputWidth;
        assertEquals(expectedOutputSize, output.getRows());
        assertEquals(1, output.getCols());

        // 2. Überprüfe, dass Output-Werte nicht alle 0 sind
        boolean hasNonZero = false;
        for (int i = 0; i < output.getRows(); i++) {
            if (Math.abs(output.get(i, 0)) > 0.0001) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Output sollte nicht-Null-Werte enthalten");

        // 3. Test Backward Pass
        Matrix error = new Matrix(output.getRows(), output.getCols());
        // Setze einige Fehler-Werte
        for (int i = 0; i < error.getRows(); i++) {
            error.set(i, 0, 0.1);
        }

        Matrix gradient = layer.backwardLayerBased(error, 0.01f);

        // Überprüfe Gradient-Dimensionen
        assertEquals(inputHeight * inputWidth, gradient.getRows());
        assertEquals(1, gradient.getCols());

        // Überprüfe, dass Gradienten nicht alle 0 sind
        hasNonZero = false;
        for (int i = 0; i < gradient.getRows(); i++) {
            if (Math.abs(gradient.get(i, 0)) > 0.0001) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Gradienten sollten nicht-Null-Werte enthalten");

    }

    @Test
    public void testBatchProcessing() {
        // Test mit einem Batch von 2 Bildern
        int numFilters = 2;
        int filterSize = 2;
        int inputHeight = 4;
        int inputWidth = 4;
        int inputChannels = 1;
        int batchSize = 2;

        ConvolutionalLayer layer = new ConvolutionalLayer(
                numFilters, filterSize, inputHeight, inputWidth, inputChannels
        );

        // Erstelle Batch-Input
        Matrix batchInput = new Matrix(inputHeight * inputWidth, batchSize);
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < inputHeight * inputWidth; i++) {
                batchInput.set(i, b, i + 1 + (b * 10));  // Verschiedene Werte für jedes Bild
            }
        }

        Matrix batchOutput = layer.forwardLayerbased(batchInput);

        // Überprüfe Batch-Output-Dimensionen
        int expectedOutputHeight = inputHeight - filterSize + 1;
        int expectedOutputWidth = inputWidth - filterSize + 1;
        int expectedOutputSize = numFilters * expectedOutputHeight * expectedOutputWidth;
        assertEquals(expectedOutputSize, batchOutput.getRows());
        assertEquals(batchSize, batchOutput.getCols());

        // Überprüfe, dass die Outputs für verschiedene Batch-Elemente unterschiedlich sind
        boolean hasOutputDifference = false;
        for (int i = 0; i < batchOutput.getRows(); i++) {
            if (Math.abs(batchOutput.get(i, 0) - batchOutput.get(i, 1)) > 0.0001) {
                hasOutputDifference = true;
                break;
            }
        }
        assertTrue(hasOutputDifference, "Batch-Outputs sollten unterschiedlich sein");
    }

    @Test
    public void testConvolutionWithFixedFilters() {
        // Test-Parameter
        int numFilters = 1;
        int filterSize = 2;
        int inputHeight = 3;
        int inputWidth = 3;
        int inputChannels = 1;

        // Layer erstellen
        ConvolutionalLayer layer = new ConvolutionalLayer(
                numFilters, filterSize, inputHeight, inputWidth, inputChannels
        );

        // Filter manuell setzen
        double[][][] filters = {{{1, 0}, {0, -1}}}; // Einfacher Sobel-ähnlicher Filter
        double[] biases = {0};
        layer.setFilters(filters);
        layer.setBiases(biases);

        // Eingabedaten
        Matrix input = new Matrix(inputHeight * inputWidth, 1);
        double[] inputData = {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        };
        for (int i = 0; i < inputData.length; i++) {
            input.set(i, 0, inputData[i]);
        }

        // Erwartete Ausgabe manuell berechnen
        // Output[0,0] = (1*1 + 2*0 + 4*0 + 5*(-1)) = 1 - 5 = -4
        // Output[0,1] = (2*1 + 3*0 + 5*0 + 6*(-1)) = 2 - 6 = -4
        // Output[1,0] = (4*1 + 5*0 + 7*0 + 8*(-1)) = 4 - 8 = -4
        // Output[1,1] = (5*1 + 6*0 + 8*0 + 9*(-1)) = 5 - 9 = -4

        double[] expectedOutputData = {-4, -4, -4, -4};
        Matrix expectedOutput = new Matrix(4, 1);
        for (int i = 0; i < expectedOutputData.length; i++) {
            expectedOutput.set(i, 0, expectedOutputData[i]);
        }

        // Forward Pass durchführen
        Matrix output = layer.forwardLayerbased(input);

        // Überprüfen, ob die Ausgabe der erwarteten entspricht
        for (int i = 0; i < expectedOutput.getRows(); i++) {
            assertEquals(expectedOutput.get(i, 0), output.get(i, 0), 0.0001);
        }
    }

}