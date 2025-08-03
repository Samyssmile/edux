package de.edux.ml.cnn.network;

import de.edux.ml.cnn.layer.Layer;

public class NetworkBuilder {
    private final NeuralNetwork network;
    
    public NetworkBuilder() {
        this.network = new NeuralNetwork();
    }
    
    public NetworkBuilder addLayer(Layer layer) {
        network.addLayer(layer);
        return this;
    }
    
    public NeuralNetwork build() {
        return network;
    }
}