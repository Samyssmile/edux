package de.edux.ml.cnn.network;

import de.edux.ml.cnn.layer.*;
import de.edux.ml.cnn.optimizer.ParameterManager;
import de.edux.ml.cnn.tensor.Tensor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements Serializable {
    private final List<Layer> layers;
    private boolean training = true;
    private transient ParameterManager parameterManager;
    
    public NeuralNetwork() {
        this.layers = new ArrayList<>();
        this.parameterManager = new ParameterManager();
    }
    
    public void addLayer(Layer layer) {
        layers.add(layer);
        registerLayerParameters(layer, layers.size() - 1);
    }
    
    private void registerLayerParameters(Layer layer, int layerIndex) {
        if (parameterManager == null) {
            parameterManager = new ParameterManager();
        }
        
        String layerName = "layer_" + layerIndex;
        
        if (layer instanceof ConvolutionalLayer) {
            ConvolutionalLayer conv = (ConvolutionalLayer) layer;
            parameterManager.registerParameter(layerName, "weights", conv.getWeights());
            if (conv.getBiases() != null) {
                parameterManager.registerParameter(layerName, "biases", conv.getBiases());
            }
        } else if (layer instanceof FullyConnectedLayer) {
            FullyConnectedLayer fc = (FullyConnectedLayer) layer;
            parameterManager.registerParameter(layerName, "weights", fc.getWeights());
            if (fc.getBiases() != null) {
                parameterManager.registerParameter(layerName, "biases", fc.getBiases());
            }
        }
    }
    
    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (Layer layer : layers) {
            layer.setTraining(training);
            current = layer.forward(current);
        }
        return current;
    }
    
    public void setTraining(boolean training) {
        this.training = training;
        for (Layer layer : layers) {
            layer.setTraining(training);
        }
    }
    
    public boolean isTraining() {
        return training;
    }
    
    public List<Layer> getLayers() {
        return new ArrayList<>(layers);
    }
    
    public int getLayerCount() {
        return layers.size();
    }
    
    public Tensor backward(Tensor gradOutput) {
        if (parameterManager == null) {
            parameterManager = new ParameterManager();
            // Re-register parameters if needed
            for (int i = 0; i < layers.size(); i++) {
                registerLayerParameters(layers.get(i), i);
            }
        }
        
        // Zero gradients first
        parameterManager.zeroGradients();
        
        // Also zero layer gradients
        for (Layer layer : layers) {
            if (layer instanceof ConvolutionalLayer) {
                ((ConvolutionalLayer) layer).zeroGradients();
            } else if (layer instanceof FullyConnectedLayer) {
                ((FullyConnectedLayer) layer).zeroGradients();
            }
        }
        
        Tensor currentGrad = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            currentGrad = layers.get(i).backward(currentGrad);
            
            // Collect gradients from parametric layers
            collectLayerGradients(layers.get(i), i);
        }
        return currentGrad;
    }
    
    private void collectLayerGradients(Layer layer, int layerIndex) {
        String layerName = "layer_" + layerIndex;
        
        if (layer instanceof ConvolutionalLayer) {
            ConvolutionalLayer conv = (ConvolutionalLayer) layer;
            if (conv.getWeightGradients() != null) {
                parameterManager.accumulateGradient(layerName, "weights", conv.getWeightGradients());
            }
            if (conv.getBiasGradients() != null) {
                parameterManager.accumulateGradient(layerName, "biases", conv.getBiasGradients());
            }
        } else if (layer instanceof FullyConnectedLayer) {
            FullyConnectedLayer fc = (FullyConnectedLayer) layer;
            if (fc.getWeightGradients() != null) {
                parameterManager.accumulateGradient(layerName, "weights", fc.getWeightGradients());
            }
            if (fc.getBiasGradients() != null) {
                parameterManager.accumulateGradient(layerName, "biases", fc.getBiasGradients());
            }
        }
    }
    
    public ParameterManager getParameterManager() {
        if (parameterManager == null) {
            parameterManager = new ParameterManager();
            for (int i = 0; i < layers.size(); i++) {
                registerLayerParameters(layers.get(i), i);
            }
        }
        return parameterManager;
    }
    
    public void zeroGradients() {
        getParameterManager().zeroGradients();
    }
    
    public void cleanup() {
        for (Layer layer : layers) {
            layer.cleanup();
        }
        if (parameterManager != null) {
            parameterManager.cleanup();
        }
    }
}