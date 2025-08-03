package de.edux.ml.cnn.training;

import de.edux.ml.cnn.data.Batch;
import de.edux.ml.cnn.data.DataLoader;
import de.edux.ml.cnn.loss.LossFunction;
import de.edux.ml.cnn.loss.LossOutput;
import de.edux.ml.cnn.network.NeuralNetwork;
import de.edux.ml.cnn.optimizer.Optimizer;
import de.edux.ml.cnn.tensor.Tensor;
import de.edux.ml.cnn.tensor.TensorPool;
import de.edux.ml.cnn.tensor.FloatTensor;

import java.util.ArrayList;
import java.util.List;

public class Trainer {
    private final NeuralNetwork network;
    private final LossFunction lossFunction;
    private final Optimizer optimizer;
    private final List<Callback> callbacks;
    
    public Trainer(NeuralNetwork network, LossFunction lossFunction, Optimizer optimizer) {
        this.network = network;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.callbacks = new ArrayList<>();
    }
    
    public void addCallback(Callback callback) {
        callbacks.add(callback);
    }
    
    public NeuralNetwork getNetwork() {
        return network;
    }
    
    public LossFunction getLossFunction() {
        return lossFunction;
    }
    
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    // Helper method for direct parameter updates
    private void updateTensor(FloatTensor params, FloatTensor grads, double learningRate) {
        float[] paramData = params.getPrimitiveData();
        float[] gradData = grads.getPrimitiveData();
        
        for (int i = 0; i < paramData.length; i++) {
            paramData[i] -= (float) (learningRate * gradData[i]);
        }
        
        params.syncFromPrimitive();
    }
    
    public void train(DataLoader trainLoader, int epochs) {
        train(trainLoader, null, epochs);
    }
    
    public void train(DataLoader trainLoader, DataLoader validationLoader, int epochs) {
        TrainerContext context = new TrainerContext(network, optimizer);
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            context.setCurrentEpoch(epoch);
            
            for (Callback callback : callbacks) {
                callback.onEpochStart(context);
            }
            
            network.setTraining(true);
            trainLoader.reset();
            
            float epochLoss = 0.0f;
            int batchCount = 0;
            int totalBatches = trainLoader.size();
            
            System.out.printf("\n=== Epoch %d/%d ===\n", epoch + 1, epochs);
            
            while (trainLoader.hasNext()) {
                Batch batch = trainLoader.next();
                context.setCurrentBatch(batchCount);
                
                // Zero gradients
                network.zeroGradients();
                
                // Forward pass
                Tensor predictions = network.forward(batch.getData());
                LossOutput lossOutput = lossFunction.compute(predictions, batch.getLabels());
                
                // Backward pass
                network.backward(lossOutput.getGradient());
                
                
                // TEMP FIX: Update parameters directly on layers
                // optimizer.update(network.getParameterManager().getParameters(), 
                //                network.getParameterManager().getGradients());
                
                // Direct layer parameter updates
                for (de.edux.ml.cnn.layer.Layer layer : network.getLayers()) {
                    if (layer instanceof de.edux.ml.cnn.layer.ConvolutionalLayer) {
                        de.edux.ml.cnn.layer.ConvolutionalLayer conv = (de.edux.ml.cnn.layer.ConvolutionalLayer) layer;
                        if (conv.getWeightGradients() != null) {
                            updateTensor(conv.getWeights(), conv.getWeightGradients(), optimizer.getLearningRate());
                        }
                        if (conv.getBiasGradients() != null) {
                            updateTensor(conv.getBiases(), conv.getBiasGradients(), optimizer.getLearningRate());
                        }
                    } else if (layer instanceof de.edux.ml.cnn.layer.FullyConnectedLayer) {
                        de.edux.ml.cnn.layer.FullyConnectedLayer fc = (de.edux.ml.cnn.layer.FullyConnectedLayer) layer;
                        if (fc.getWeightGradients() != null) {
                            updateTensor(fc.getWeights(), fc.getWeightGradients(), optimizer.getLearningRate());
                        }
                        if (fc.getBiasGradients() != null) {
                            updateTensor(fc.getBiases(), fc.getBiasGradients(), optimizer.getLearningRate());
                        }
                    }
                }
                
                float batchLoss = lossOutput.getLoss();
                epochLoss += batchLoss;
                batchCount++;
                
                context.setCurrentLoss(batchLoss);
                
                // Clean batch progress
                int batchesLeft = totalBatches - batchCount;
                if (batchCount % 10 == 0 || batchesLeft == 0) {
                    System.out.printf("\rBatch %d/%d | Loss: %.4f | Remaining: %d batches", 
                        batchCount, totalBatches, batchLoss, batchesLeft);
                    System.out.flush();
                }
                
                for (Callback callback : callbacks) {
                    callback.onBatchEnd(context);
                }
            }
            
            float avgLoss = epochLoss / batchCount;
            context.setCurrentLoss(avgLoss);
            
            // Calculate accuracy on validation set if provided, otherwise on training set
            float accuracy = 0.0f;
            if (validationLoader != null) {
                accuracy = evaluate(validationLoader);
            } else {
                accuracy = evaluate(trainLoader);
            }
            context.setCurrentAccuracy(accuracy);
            
            // Complete epoch metrics
            System.out.printf("\nEpoch %d/%d Results:\n", epoch + 1, epochs);
            System.out.printf("  • Average Loss: %.6f\n", avgLoss);
            System.out.printf("  • Accuracy: %.2f%%\n", accuracy * 100);
            System.out.printf("  • Learning Rate: %.6f\n", optimizer.getLearningRate());
            System.out.printf("  • Batches Processed: %d\n", batchCount);
            
            long epochTime = System.currentTimeMillis();
            if (epoch == 0) {
                System.out.printf("  • Time: Training started\n");
            } else {
                System.out.printf("  • Status: Epoch completed\n");
            }
            
            for (Callback callback : callbacks) {
                callback.onEpochEnd(context);
            }
        }
    }
    
    public float evaluate(DataLoader testLoader) {
        network.setTraining(false);
        testLoader.reset();
        
        float totalLoss = 0.0f;
        int correct = 0;
        int total = 0;
        
        while (testLoader.hasNext()) {
            Batch batch = testLoader.next();
            Tensor predictions = network.forward(batch.getData());
            LossOutput lossOutput = lossFunction.compute(predictions, batch.getLabels());
            
            totalLoss += lossOutput.getLoss();
            
            // Use primitive arrays for faster evaluation
            FloatTensor predTensor = (FloatTensor) predictions;
            FloatTensor labelTensor = (FloatTensor) batch.getLabels();
            
            float[] predData = predTensor.getPrimitiveData();
            float[] labelData = labelTensor.getPrimitiveData();
            
            int batchSize = predictions.getShape()[0];
            int numClasses = predictions.getShape()[1];
            
            for (int i = 0; i < batchSize; i++) {
                int predClass = 0;
                int trueClass = 0;
                float maxPred = Float.NEGATIVE_INFINITY;
                
                for (int j = 0; j < numClasses; j++) {
                    int idx = i * numClasses + j;
                    if (predData[idx] > maxPred) {
                        maxPred = predData[idx];
                        predClass = j;
                    }
                    if (labelData[idx] > 0.5f) {
                        trueClass = j;
                    }
                }
                
                if (predClass == trueClass) {
                    correct++;
                }
                total++;
            }
            
            // Clean up temporary tensors if they were created by pooling
            if (predictions instanceof FloatTensor) {
                // Only release if it was created by pooling (check by comparing with cache)
                // For now, we'll rely on GC but could add more sophisticated tracking
            }
        }
        
        return (float) correct / total;
    }
    
    /**
     * Clean up resources and clear tensor pools
     */
    public void cleanup() {
        TensorPool.clear();
        System.gc(); // Suggest garbage collection
    }
}