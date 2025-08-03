package de.edux.ml.cnn.optimizer;

import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;

import java.util.Map;

public class SGD implements Optimizer {
    private double learningRate;
    
    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }
    
    @Override
    public void update(Map<Parameter, Tensor> params, Map<Parameter, Tensor> grads) {
        for (Map.Entry<Parameter, Tensor> entry : params.entrySet()) {
            Parameter param = entry.getKey();
            Tensor paramTensor = entry.getValue();
            Tensor gradTensor = grads.get(param);
            
            if (gradTensor != null && paramTensor instanceof FloatTensor && gradTensor instanceof FloatTensor) {
                FloatTensor paramFloatTensor = (FloatTensor) paramTensor;
                FloatTensor gradFloatTensor = (FloatTensor) gradTensor;
                
                float[] paramData = paramFloatTensor.getPrimitiveData();
                float[] gradData = gradFloatTensor.getPrimitiveData();
                
                for (int i = 0; i < paramData.length; i++) {
                    paramData[i] -= (float) (learningRate * gradData[i]);
                }
                
                // Sync changes back to boxed array
                paramFloatTensor.syncFromPrimitive();
            }
        }
    }
    
    @Override
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    @Override
    public double getLearningRate() {
        return learningRate;
    }
}