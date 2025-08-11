package de.edux.ml.cnn.training;

import de.edux.ml.cnn.network.NeuralNetwork;
import de.edux.ml.cnn.optimizer.Optimizer;

public class TrainerContext {
    private final NeuralNetwork network;
    private final Optimizer optimizer;
    private int currentEpoch;
    private int currentBatch;
    private float currentLoss;
    private float currentAccuracy;
    
    public TrainerContext(NeuralNetwork network, Optimizer optimizer) {
        this.network = network;
        this.optimizer = optimizer;
    }
    
    public NeuralNetwork getNetwork() {
        return network;
    }
    
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    public int getCurrentEpoch() {
        return currentEpoch;
    }
    
    public void setCurrentEpoch(int currentEpoch) {
        this.currentEpoch = currentEpoch;
    }
    
    public int getCurrentBatch() {
        return currentBatch;
    }
    
    public void setCurrentBatch(int currentBatch) {
        this.currentBatch = currentBatch;
    }
    
    public float getCurrentLoss() {
        return currentLoss;
    }
    
    public void setCurrentLoss(float currentLoss) {
        this.currentLoss = currentLoss;
    }
    
    public float getCurrentAccuracy() {
        return currentAccuracy;
    }
    
    public void setCurrentAccuracy(float currentAccuracy) {
        this.currentAccuracy = currentAccuracy;
    }
}