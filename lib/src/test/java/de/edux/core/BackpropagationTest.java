package de.edux.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import de.edux.ml.mlp.core.network.loss.LossFunctions;
import de.edux.ml.mlp.core.network.optimizer.Approximator;
import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;
import org.junit.jupiter.api.Test;

public class BackpropagationTest {
    private final Random random = new Random();

    @Test
    void shouldBackpropagate() {

        interface NeuralNetwork {
            Matrix apply(Matrix m);
        }
        final int inputRows = 4;
        final int cols = 5;
        final int outputRows = 4;

        Matrix input =new Matrix(inputRows, cols, i->random.nextGaussian());
        Matrix expected = new Matrix(outputRows, cols, i->0);

        Matrix weights = new Matrix(outputRows, inputRows, i->random.nextGaussian());
        Matrix biases = new Matrix(outputRows, 1, i->random.nextGaussian());

        for(int col =0; col <cols; col++){
            int randowmRow = random.nextInt(outputRows);
            expected.set(randowmRow, col, 1);
        }

        NeuralNetwork neuralNet = (m)->{
            Matrix out = m.relu(); //input
            out = weights.multiply(out).add(biases); // Dense
            out = out.softmax(); // Softmax
            return out;
        };

        Matrix softmaxOutput = neuralNet.apply(input);

        Matrix approximatedResult = Approximator.gradient(input, in-> {
            Matrix out = neuralNet.apply(in);
            return LossFunctions.crossEntropy(expected, out);
        });

        Matrix calculatedResult = softmaxOutput.subtract(expected); //Softmax backward
        calculatedResult = weights.transpose().multiply(calculatedResult);
        calculatedResult = calculatedResult.reluDerivative(input);

        System.out.println("Approximated Result");
        System.out.println(approximatedResult);
        System.out.println("Backpropagated Result");
        System.out.println(calculatedResult);

        assertEquals(approximatedResult, calculatedResult);
    }
}
