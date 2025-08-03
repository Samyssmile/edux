package de.edux.ml.cnn.inference;

import de.edux.ml.cnn.activation.Softmax;
import de.edux.ml.cnn.network.NeuralNetwork;
import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.tensor.Tensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Predictor {
    private final NeuralNetwork network;
    private final Softmax softmax;
    
    public Predictor(NeuralNetwork network) {
        if (network == null) {
            throw new IllegalArgumentException("Neural network cannot be null");
        }
        this.network = network;
        this.network.setTraining(false);
        this.softmax = new Softmax();
    }
    
    public int predict(String imagePath) {
        FloatTensor imageTensor = loadImage(imagePath);
        FloatTensor batchTensor = addBatchDimension(imageTensor);
        
        Tensor logits = network.forward(batchTensor);
        Tensor predictions = softmax.apply(logits);
        Float[] predData = (Float[]) predictions.getData();
        
        int predictedClass = 0;
        float maxProb = Float.NEGATIVE_INFINITY;
        
        for (int i = 0; i < predData.length; i++) {
            if (predData[i] > maxProb) {
                maxProb = predData[i];
                predictedClass = i;
            }
        }
        
        return predictedClass;
    }
    
    public float[] predictProbabilities(String imagePath) {
        FloatTensor imageTensor = loadImage(imagePath);
        FloatTensor batchTensor = addBatchDimension(imageTensor);
        
        Tensor logits = network.forward(batchTensor);
        Tensor predictions = softmax.apply(logits);
        Float[] predData = (Float[]) predictions.getData();
        
        float[] probabilities = new float[predData.length];
        for (int i = 0; i < predData.length; i++) {
            probabilities[i] = predData[i];
        }
        
        return probabilities;
    }
    
    public int[] predictBatch(Tensor batchImages) {
        Tensor logits = network.forward(batchImages);
        Tensor predictions = softmax.apply(logits);
        Float[] predData = (Float[]) predictions.getData();
        
        int[] shape = predictions.getShape();
        int batchSize = shape[0];
        int numClasses = shape[1];
        
        int[] predictedClasses = new int[batchSize];
        
        for (int b = 0; b < batchSize; b++) {
            int predictedClass = 0;
            float maxProb = Float.NEGATIVE_INFINITY;
            
            for (int c = 0; c < numClasses; c++) {
                int idx = b * numClasses + c;
                if (predData[idx] > maxProb) {
                    maxProb = predData[idx];
                    predictedClass = c;
                }
            }
            
            predictedClasses[b] = predictedClass;
        }
        
        return predictedClasses;
    }
    
    private FloatTensor loadImage(String imagePath) {
        try {
            BufferedImage image = ImageIO.read(new File(imagePath));
            int width = image.getWidth();
            int height = image.getHeight();
            
            // Check if image is grayscale by examining if R==G==B for a few pixels
            boolean isGrayscale = true;
            for (int y = 0; y < Math.min(5, height) && isGrayscale; y++) {
                for (int x = 0; x < Math.min(5, width) && isGrayscale; x++) {
                    int rgb = image.getRGB(x, y);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    if (r != g || g != b) {
                        isGrayscale = false;
                    }
                }
            }
            
            if (isGrayscale) {
                // Single channel grayscale
                float[] imageData = new float[height * width];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int rgb = image.getRGB(x, y);
                        int gray = (rgb >> 16) & 0xFF; // Use red channel (all channels are equal)
                        imageData[y * width + x] = gray / 255.0f;
                    }
                }
                return FloatTensor.fromArray(imageData, 1, height, width);
            } else {
                // RGB image - 3 channels
                float[] imageData = new float[3 * height * width];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int rgb = image.getRGB(x, y);
                        int r = (rgb >> 16) & 0xFF;
                        int g = (rgb >> 8) & 0xFF;
                        int b = rgb & 0xFF;
                        
                        int idx = y * width + x;
                        imageData[idx] = r / 255.0f;                           // Red channel
                        imageData[idx + width * height] = g / 255.0f;          // Green channel  
                        imageData[idx + 2 * width * height] = b / 255.0f;      // Blue channel
                    }
                }
                return FloatTensor.fromArray(imageData, 3, height, width);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load image: " + imagePath, e);
        }
    }
    
    private FloatTensor addBatchDimension(FloatTensor imageTensor) {
        int[] originalShape = imageTensor.getShape();
        int[] batchShape = new int[originalShape.length + 1];
        batchShape[0] = 1;
        System.arraycopy(originalShape, 0, batchShape, 1, originalShape.length);
        
        return imageTensor.reshape(batchShape);
    }
}