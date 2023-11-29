package de.edux.ml.cnn;

import de.edux.ml.cnn.layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private List<Layer> layers = new ArrayList<>();

    public NetworkBuilder addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public Network build() {
        return new Network(this.layers);
    }

}
