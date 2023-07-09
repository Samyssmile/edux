package de.nexent.edux.nn.network;

import de.nexent.edux.functions.activation.ActivationFunction;
import de.nexent.edux.functions.loss.LossFunction;
import de.nexent.edux.ml.nn.config.Configuration;
import de.nexent.edux.ml.nn.network.MultilayerPerceptron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class MultilayerPerceptronTest {

    private MultilayerPerceptron mlp;
    private Configuration config;

    @BeforeEach
    public void setUp() {
        double[][] inputs = {{0.1, 0.2, 0.3}};
        double[][] targets = {{0, 1, 0}};
        double[][] testInputs = {{0.2, 0.3, 0.4}};
        double[][] testTargets = {{0, 0, 1}};
        config = new Configuration(3, List.of(5), 3, 0.01, 1000, ActivationFunction.LEAKY_RELU, ActivationFunction.SOFTMAX, LossFunction.CATEGORICAL_CROSS_ENTROPY);

        mlp = new MultilayerPerceptron(inputs, targets, testInputs, testTargets, config);
    }

    @Test
    public void shouldNotNull() {
        double[] output = mlp.predict(new double[]{0.1, 0.2, 0.3});
        assertNotNull(output);
        assertEquals(config.outputSize(), output.length);
    }

    @Test
    public void testTrainAndEvaluate() {
        mlp.train();
        double accuracy = mlp.evaluate(new double[][]{{0.2, 0.3, 0.4}}, new double[][]{{0, 0, 1}});
        assertNotNull(accuracy);
    }
}
