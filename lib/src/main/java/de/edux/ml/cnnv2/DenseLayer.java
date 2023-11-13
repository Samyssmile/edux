package de.edux.ml.cnnv2;

import de.edux.ml.cnn.layers.Layer;
import de.edux.ml.cnn.math.Matrix;
import de.edux.ml.cnn.math.Matrix3D;
import de.edux.ml.cnn.math.Vector;
public class DenseLayer implements Layer {

    private Matrix3D weights;
    private Matrix3D biases;
    private Matrix3D input;
    private Matrix3D outputGradient;

    public DenseLayer(int inputSize, int outputSize, int depth) {
        this.weights =  Matrix3D.random(depth, outputSize, inputSize);
        this.biases = new Matrix3D(depth, outputSize, 1);
        // Initialisieren Sie weights und biases hier (zufällig oder mit einer spezifischen Initialisierungsstrategie)
    }

    @Override
    public Matrix3D forward(Matrix3D input) {
        // Speichern Sie die Eingabe für die Verwendung im Backward-Pass
        this.input = input;

        // Führen Sie die lineare Transformation durch: output = input * weights + biases
        Matrix3D output = input.dot(this.weights).add(this.biases);

        return output;
    }


    @Override
    public Matrix3D backward(Matrix3D outputGradient, double learningRate) {
        // Berechnen Sie den Gradienten für die Gewichte (dW) und die Biases (db)
        Matrix3D dW = this.input.transpose().dot(outputGradient);
        Matrix3D db = outputGradient;

        // Aktualisieren Sie die Gewichte und Biases
        this.weights = this.weights.subtract(dW.multiply(learningRate));
        this.biases = this.biases.subtract(db.multiply(learningRate));

        // Berechnen Sie den Gradienten für die nächste Schicht
        Matrix3D inputGradient = outputGradient.dot(this.weights.transpose());

        return inputGradient;
    }


}
