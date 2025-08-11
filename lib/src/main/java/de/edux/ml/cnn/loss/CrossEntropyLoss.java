package de.edux.ml.cnn.loss;

import de.edux.ml.cnn.activation.Softmax;
import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;

public class CrossEntropyLoss implements LossFunction {
    private final Softmax softmax;
    private final float epsilon = 1e-15f;
    
    public CrossEntropyLoss() {
        this.softmax = new Softmax();
    }
    
    @Override
    public LossOutput compute(Tensor predictions, Tensor labels) {
        int[] predShape = predictions.getShape();
        int[] labelShape = labels.getShape();
        
        if (predShape.length != 2 || labelShape.length != 2) {
            throw new IllegalArgumentException("Predictions and labels must be 2D");
        }
        
        if (predShape[0] != labelShape[0]) {
            throw new IllegalArgumentException("Batch sizes must match");
        }
        
        int batch = predShape[0];
        int numClasses = predShape[1];
        
        Tensor softmaxOutput = softmax.apply(predictions);
        float[] softmaxData = ((FloatTensor) softmaxOutput).getPrimitiveData();
        float[] labelData = ((FloatTensor) labels).getPrimitiveData();
        
        float totalLoss = 0.0f;
        FloatTensor gradient = FloatTensor.zeros(predShape);
        float[] gradData = gradient.getPrimitiveData();
        
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < numClasses; c++) {
                int idx = b * numClasses + c;
                float prob = Math.max(epsilon, Math.min(1.0f - epsilon, softmaxData[idx]));
                float label = labelData[idx];
                
                if (label > 0) {
                    totalLoss -= label * (float) Math.log(prob);
                }
                
                gradData[idx] = (prob - label) / batch;
            }
        }
        
        // Sync primitive data back to boxed array
        gradient.syncFromPrimitive();
        
        return new LossOutput(totalLoss / batch, gradient);
    }
}