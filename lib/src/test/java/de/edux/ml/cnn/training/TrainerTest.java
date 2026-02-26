package de.edux.ml.cnn.training;

import de.edux.ml.cnn.data.Batch;
import de.edux.ml.cnn.data.DataLoader;
import de.edux.ml.cnn.loss.LossFunction;
import de.edux.ml.cnn.loss.LossOutput;
import de.edux.ml.cnn.network.NeuralNetwork;
import de.edux.ml.cnn.optimizer.Optimizer;
import de.edux.ml.cnn.optimizer.Parameter;
import de.edux.ml.cnn.optimizer.SGD;
import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class TrainerTest {

    @Test
    void shouldUseOptimizerForParameterUpdate() {
        TestNetwork network = new TestNetwork();
        RecordingOptimizer optimizer = new RecordingOptimizer(0.1);
        LossFunction lossFunction = (predictions, labels) ->
            new LossOutput(0.1f, FloatTensor.zeros(predictions.getShape()));
        Trainer trainer = new Trainer(network, lossFunction, optimizer);

        FloatTensor data = FloatTensor.fromArray(new float[]{0.9f, 0.1f}, 1, 2);
        FloatTensor labels = FloatTensor.fromArray(new float[]{1.0f, 0.0f}, 1, 2);
        DataLoader loader = new SingleBatchDataLoader(new Batch(data, labels));

        float initialWeight = network.getWeights().getPrimitiveData()[0];

        trainer.train(loader, 1);

        float updatedWeight = network.getWeights().getPrimitiveData()[0];

        assertEquals(1, optimizer.getUpdateCalls());
        assertNotEquals(initialWeight, updatedWeight);
        assertEquals(0.95f, updatedWeight, 1e-6f);
    }

    private static final class TestNetwork extends NeuralNetwork {
        private static final String LAYER = "layer_0";
        private static final String WEIGHTS = "weights";

        private final FloatTensor weights = FloatTensor.fromArray(new float[]{1.0f}, 1);
        private final FloatTensor weightGradient = FloatTensor.fromArray(new float[]{0.5f}, 1);

        private TestNetwork() {
            getParameterManager().registerParameter(LAYER, WEIGHTS, weights);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }

        @Override
        public Tensor backward(Tensor gradOutput) {
            getParameterManager().zeroGradients();
            getParameterManager().accumulateGradient(LAYER, WEIGHTS, weightGradient);
            return gradOutput;
        }

        private FloatTensor getWeights() {
            return weights;
        }
    }

    private static final class SingleBatchDataLoader implements DataLoader {
        private final Batch batch;
        private boolean consumed = false;

        private SingleBatchDataLoader(Batch batch) {
            this.batch = batch;
        }

        @Override
        public void shuffle() {
            // no-op for deterministic test data
        }

        @Override
        public void reset() {
            consumed = false;
        }

        @Override
        public int size() {
            return 1;
        }

        @Override
        public boolean hasNext() {
            return !consumed;
        }

        @Override
        public Batch next() {
            if (consumed) {
                throw new NoSuchElementException("No batch left");
            }
            consumed = true;
            return batch;
        }
    }

    private static final class RecordingOptimizer implements Optimizer {
        private final SGD delegate;
        private int updateCalls;

        private RecordingOptimizer(double learningRate) {
            this.delegate = new SGD(learningRate);
        }

        @Override
        public void update(Map<Parameter, Tensor> params, Map<Parameter, Tensor> grads) {
            updateCalls++;
            delegate.update(params, grads);
        }

        @Override
        public void setLearningRate(double learningRate) {
            delegate.setLearningRate(learningRate);
        }

        @Override
        public double getLearningRate() {
            return delegate.getLearningRate();
        }

        private int getUpdateCalls() {
            return updateCalls;
        }
    }
}
