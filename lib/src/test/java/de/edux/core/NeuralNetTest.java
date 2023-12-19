package de.edux.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import de.edux.ml.mlp.core.network.Engine;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loss.LossFunctions;
import de.edux.ml.mlp.core.network.optimizer.Approximator;
import de.edux.ml.mlp.core.tensor.Matrix;
import de.edux.ml.mlp.util.Util;
import java.util.Random;
import org.junit.jupiter.api.Test;

public class NeuralNetTest {

    private final Random random = new Random();


     @Test
    void testWeightGradient() {
        int inputRows = 4;
        int outputRows = 5;
        Matrix weights = new Matrix(outputRows, inputRows, i -> random.nextGaussian());
        Matrix input = Util.generateInputMatrix(inputRows, 1);
        Matrix expected = Util.generateExpectedMatrix(outputRows, 1);
        Matrix output = weights.multiply(input).softmax();

        Matrix loss = LossFunctions.crossEntropy(expected, output);

        Matrix calculatedGradient = output.apply((index, value) -> value - expected.get(index));

        Matrix calculatedWeightGradients = calculatedGradient.multiply(input.transpose());

        Matrix approximatedWeightGradients = Approximator.weightGradient(weights, in -> {
            Matrix out = in.multiply(input).softmax();
            return LossFunctions.crossEntropy(expected, out);
        });

        calculatedWeightGradients.setTolerance(0.01);
        assertEquals(approximatedWeightGradients, calculatedWeightGradients);
    }

    @Test
    void testEngineLayerbased() {
        int rows = 5;
        int cols = 6;
        int outputRows = 4;

        Engine engine = new Engine(10);
        engine.addLayer(new DenseLayer(5, 8));
        engine.addLayer(new ReLuLayer());
        engine.addLayer(new DenseLayer(8, 5));
        engine.addLayer(new ReLuLayer());
        engine.addLayer(new DenseLayer(5, 4));
        engine.addLayer(new SoftmaxLayer());

        Matrix input = Util.generateInputMatrix(rows, cols);
        Matrix expected = Util.generateTrainableExpectedMatrix(outputRows, input);

        //Forward pass
        Matrix softmaxOutput = engine.forwardLayerbased(input);

        //Loss function
        Matrix approximatedError = Approximator.gradient(input, in -> {
            Matrix forwardResult = engine.forwardLayerbased(in);
            return LossFunctions.crossEntropy(expected, forwardResult);
        });


        //Backward pass
        Matrix calculatedError = engine.backwardLayerBased(expected, 0.01f);
        System.out.println("Approximated Error");
        System.out.println(approximatedError);
        System.out.println("Calculated Error");
        System.out.println(calculatedError);

        calculatedError.setTolerance(0.0001);

        assertEquals(approximatedError, calculatedError);

    }


}
