package de.edux.ml.cnn.optimizer;

public class Parameter {
    private final String name;
    private final String layerName;
    
    public Parameter(String name, String layerName) {
        this.name = name;
        this.layerName = layerName;
    }
    
    public String getName() {
        return name;
    }
    
    public String getLayerName() {
        return layerName;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Parameter parameter = (Parameter) obj;
        return name.equals(parameter.name) && layerName.equals(parameter.layerName);
    }
    
    @Override
    public int hashCode() {
        return name.hashCode() + layerName.hashCode();
    }
    
    @Override
    public String toString() {
        return layerName + "." + name;
    }
}