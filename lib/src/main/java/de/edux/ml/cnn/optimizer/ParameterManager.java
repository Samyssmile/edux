package de.edux.ml.cnn.optimizer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

public class ParameterManager {
    private final Map<Parameter, FloatTensor> parameters;
    private final Map<Parameter, FloatTensor> gradients;
    
    public ParameterManager() {
        this.parameters = new HashMap<>();
        this.gradients = new HashMap<>();
    }
    
    public void registerParameter(String layerName, String paramName, FloatTensor tensor) {
        Parameter param = new Parameter(paramName, layerName);
        parameters.put(param, tensor);
        
        int[] shape = tensor.getShape();
        FloatTensor gradient = FloatTensor.zeros(shape);
        gradients.put(param, gradient);
    }
    
    public void accumulateGradient(String layerName, String paramName, FloatTensor gradient) {
        Parameter param = new Parameter(paramName, layerName);
        FloatTensor existingGrad = gradients.get(param);
        
        if (existingGrad != null) {
            float[] existingData = existingGrad.getPrimitiveData();
            float[] newGradData = gradient.getPrimitiveData();
            
            for (int i = 0; i < existingData.length; i++) {
                existingData[i] += newGradData[i];
            }
            existingGrad.syncFromPrimitive();
        }
    }
    
    public void zeroGradients() {
        for (FloatTensor gradient : gradients.values()) {
            float[] gradData = gradient.getPrimitiveData();
            for (int i = 0; i < gradData.length; i++) {
                gradData[i] = 0.0f;
            }
            gradient.syncFromPrimitive();
        }
    }
    
    public Map<Parameter, Tensor> getParameters() {
        Map<Parameter, Tensor> result = new HashMap<>();
        for (Map.Entry<Parameter, FloatTensor> entry : parameters.entrySet()) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }
    
    public Map<Parameter, Tensor> getGradients() {
        Map<Parameter, Tensor> result = new HashMap<>(); 
        for (Map.Entry<Parameter, FloatTensor> entry : gradients.entrySet()) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }
    
    public FloatTensor getParameter(String layerName, String paramName) {
        Parameter param = new Parameter(paramName, layerName);
        return parameters.get(param);
    }
    
    public FloatTensor getGradient(String layerName, String paramName) {
        Parameter param = new Parameter(paramName, layerName);
        return gradients.get(param);
    }
    
    public void cleanup() {
        for (FloatTensor gradient : gradients.values()) {
            gradient.dispose();
        }
        gradients.clear();
        parameters.clear();
    }
}