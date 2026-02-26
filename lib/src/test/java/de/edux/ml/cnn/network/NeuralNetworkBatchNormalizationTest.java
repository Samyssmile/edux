package de.edux.ml.cnn.network;

import de.edux.ml.cnn.layer.BatchNormalizationLayer;
import de.edux.ml.cnn.optimizer.ParameterManager;
import de.edux.ml.cnn.optimizer.SGD;
import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralNetworkBatchNormalizationTest {

    @Test
    void shouldRegisterBatchNormalizationParameters() {
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new BatchNormalizationLayer(3));

        ParameterManager parameterManager = network.getParameterManager();

        assertNotNull(parameterManager.getParameter("layer_0", "gamma"));
        assertNotNull(parameterManager.getParameter("layer_0", "beta"));
    }

    @Test
    void shouldUpdateBatchNormalizationParametersWithOptimizer() {
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new BatchNormalizationLayer(3));

        FloatTensor input = FloatTensor.fromArray(
            new float[]{1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f},
            2, 3
        );
        Tensor output = network.forward(input);
        network.backward(FloatTensor.ones(output.getShape()));

        ParameterManager parameterManager = network.getParameterManager();
        FloatTensor beta = parameterManager.getParameter("layer_0", "beta");
        FloatTensor betaGradient = parameterManager.getGradient("layer_0", "beta");

        assertNotNull(betaGradient);
        assertTrue(hasNonZero(betaGradient));

        float[] betaBefore = beta.getDataArrayPrimitive();
        new SGD(0.1).update(parameterManager.getParameters(), parameterManager.getGradients());
        float[] betaAfter = beta.getDataArrayPrimitive();

        assertTrue(changed(betaBefore, betaAfter));
    }

    private boolean hasNonZero(FloatTensor tensor) {
        for (float value : tensor.getPrimitiveData()) {
            if (Math.abs(value) > 1e-7f) {
                return true;
            }
        }
        return false;
    }

    private boolean changed(float[] before, float[] after) {
        for (int i = 0; i < before.length; i++) {
            if (Math.abs(before[i] - after[i]) > 1e-7f) {
                return true;
            }
        }
        return false;
    }
}
