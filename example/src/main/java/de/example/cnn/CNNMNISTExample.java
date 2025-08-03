package de.example.cnn;

import de.edux.ml.cnn.data.MNISTDataLoader;
import de.edux.ml.cnn.inference.Predictor;
import de.edux.ml.cnn.layer.*;
import de.edux.ml.cnn.loss.CrossEntropyLoss;
import de.edux.ml.cnn.network.NetworkBuilder;
import de.edux.ml.cnn.network.NeuralNetwork;
import de.edux.ml.cnn.optimizer.SGD;
import de.edux.ml.cnn.tensor.FloatTensor;
import de.edux.ml.cnn.training.Callback;
import de.edux.ml.cnn.training.Trainer;
import de.edux.ml.cnn.training.TrainerContext;

import java.io.*;

public class CNNMNISTExample {
    public static void main(String[] args) {
        System.out.println("Starting MNIST CNN Training with EDUX Library");
        System.out.println("==============================================");
        
        String datasetRoot = "edux/example/datasets/mnist-pngs";
        String trainCsv = datasetRoot + "/train.csv";
        String testCsv = datasetRoot + "/test.csv";
        String modelPath = "edux-cnn-model.ser";
        
        int batchSize = 128;
        int epochs = 3;
        double learningRate = 0.001;
        
        System.out.println("Training Configuration:");
        System.out.println("- Batch Size: " + batchSize);
        System.out.println("- Epochs: " + epochs);
        System.out.println("- Learning Rate: " + learningRate);
        System.out.println("- Model will be saved as: " + modelPath);
        System.out.println();
        
        System.out.println("Loading dataset...");
        MNISTDataLoader trainLoader = new MNISTDataLoader(trainCsv, datasetRoot, batchSize, 1);
        MNISTDataLoader testLoader = new MNISTDataLoader(testCsv, datasetRoot, batchSize, 1);
        
        System.out.println("Building CNN Architecture with EDUX:");
        System.out.println("Architecture Details:");
        System.out.println("- Input: 28x28x1 (grayscale images)");
        System.out.println("- Conv1: 1→32 filters, 3x3 kernel, ReLU");
        System.out.println("- Conv2: 32→32 filters, 3x3 kernel, ReLU");
        System.out.println("- MaxPool: 2x2 (28x28 → 14x14)");
        System.out.println("- Conv3: 32→64 filters, 3x3 kernel, ReLU");
        System.out.println("- Conv4: 64→64 filters, 3x3 kernel, ReLU");
        System.out.println("- MaxPool: 2x2 (14x14 → 7x7)");
        System.out.println("- Conv5: 64→128 filters, 3x3 kernel, ReLU");
        System.out.println("- MaxPool: 2x2 (7x7 → 3x3)");
        System.out.println("- Flatten: 128×3×3 = 1152 features");
        System.out.println("- FC1: 1152→256, ReLU");
        System.out.println("- FC2: 256→128, ReLU");
        System.out.println("- FC3: 128→10 (output classes)");
        System.out.println();
        
        NeuralNetwork cnn = new NetworkBuilder()
                .addLayer(new ConvolutionalLayer(1, 32, 3, 1, 1))
                .addLayer(new ReLuLayer())
                .addLayer(new ConvolutionalLayer(32, 32, 3, 1, 1))
                .addLayer(new ReLuLayer())
                .addLayer(new PoolingLayer(PoolingLayer.PoolingType.MAX, 2))
                
                .addLayer(new ConvolutionalLayer(32, 64, 3, 1, 1))
                .addLayer(new ReLuLayer())
                .addLayer(new ConvolutionalLayer(64, 64, 3, 1, 1))
                .addLayer(new ReLuLayer())
                .addLayer(new PoolingLayer(PoolingLayer.PoolingType.MAX, 2))
                
                .addLayer(new ConvolutionalLayer(64, 128, 3, 1, 1))
                .addLayer(new ReLuLayer())
                .addLayer(new PoolingLayer(PoolingLayer.PoolingType.MAX, 2))
                
                .addLayer(new FlattenLayer())
                .addLayer(new FullyConnectedLayer(128 * 3 * 3, 256))
                .addLayer(new ReLuLayer())
                .addLayer(new FullyConnectedLayer(256, 128))
                .addLayer(new ReLuLayer())
                .addLayer(new FullyConnectedLayer(128, 10))
                .build();
        
        System.out.println("Network built successfully with EDUX!");
        System.out.printf("Total layers: %d\n", cnn.getLayerCount());
        System.out.println();
        
        System.out.println("Setting up training...");
        CrossEntropyLoss lossFunction = new CrossEntropyLoss();
        SGD optimizer = new SGD(learningRate);
        Trainer trainer = new Trainer(cnn, lossFunction, optimizer);
        
        trainer.addCallback(new Callback() {
            private long epochStartTime;
            private double bestLoss = Double.MAX_VALUE;
            
            @Override
            public void onEpochStart(TrainerContext ctx) {
                epochStartTime = System.currentTimeMillis();
                System.out.printf("\n🔄 Starting Epoch %d/%d with EDUX CNN\n", ctx.getCurrentEpoch() + 1, epochs);
                System.out.println("─────────────────────────────────────────");
            }
            
            @Override
            public void onEpochEnd(TrainerContext ctx) {
                long epochTime = System.currentTimeMillis() - epochStartTime;
                double currentLoss = ctx.getCurrentLoss();
                
                System.out.printf("\n✅ Epoch %d completed in %.1f seconds\n", 
                    ctx.getCurrentEpoch() + 1, epochTime / 1000.0);
                System.out.printf("   Final Loss: %.6f", currentLoss);
                
                if (currentLoss < bestLoss) {
                    bestLoss = currentLoss;
                    System.out.print(" 🌟 (New Best!)");
                }
                System.out.println();
                
                if (currentLoss < 0.1) {
                    System.out.println("   📈 Excellent convergence!");
                } else if (currentLoss < 0.3) {
                    System.out.println("   📊 Good progress!");
                } else if (currentLoss < 0.5) {
                    System.out.println("   📉 Learning steadily...");
                }
            }
            
            @Override  
            public void onBatchEnd(TrainerContext ctx) {
                if (ctx.getCurrentBatch() % 50 == 0) {
                    System.out.printf("   Batch %3d - Loss: %.4f\n", 
                        ctx.getCurrentBatch(), ctx.getCurrentLoss());
                }
            }
        });
        
        System.out.println("🚀 Starting training with EDUX CNN...");
        
        trainLoader.shuffle();
        long trainingStartTime = System.currentTimeMillis();
        trainer.train(trainLoader, epochs);
        long trainingEndTime = System.currentTimeMillis();
        
        double totalTrainingTime = (trainingEndTime - trainingStartTime) / 1000.0;
        System.out.printf("\n🎉 Training completed in %.1f seconds (%.1f minutes)\n", 
            totalTrainingTime, totalTrainingTime / 60.0);
        
        System.out.println("\n📊 Evaluating EDUX CNN model...");
        float accuracy = trainer.evaluate(testLoader);
        System.out.printf("🎯 Test Accuracy: %.2f%%\n", accuracy * 100);
        
        if (accuracy > 0.97) {
            System.out.println("🏆 Outstanding performance! EDUX CNN model is excellent.");
        } else if (accuracy > 0.95) {
            System.out.println("🥇 Great performance! EDUX CNN model is very good.");
        } else if (accuracy > 0.93) {
            System.out.println("🥈 Good performance! EDUX CNN showing improvement.");
        } else {
            System.out.println("🥉 Moderate performance. Consider further improvements.");
        }
        
        System.out.println("\n🔍 Making sample predictions with EDUX CNN...");
        Predictor predictor = new Predictor(cnn);
        
        testLoader.reset();
        if (testLoader.hasNext()) {
            var batch = testLoader.next();
            int[] predictions = predictor.predictBatch(batch.getData());
            
            FloatTensor labels = (FloatTensor) batch.getLabels();
            float[] labelData = labels.getPrimitiveData();
            int testBatchSize = labels.getShape()[0];
            int numClasses = labels.getShape()[1];
            
            System.out.println("Sample predictions vs actual labels:");
            int correctInSample = 0;
            for (int i = 0; i < Math.min(10, testBatchSize); i++) {
                int actualClass = -1;
                for (int j = 0; j < numClasses; j++) {
                    int idx = i * numClasses + j;
                    if (labelData[idx] > 0.5f) {
                        actualClass = j;
                        break;
                    }
                }
                
                boolean correct = (predictions[i] == actualClass);
                if (correct) correctInSample++;
                
                System.out.printf("  Sample %2d: Predicted=%d, Actual=%d %s\n", 
                    i + 1, predictions[i], actualClass, correct ? "✅" : "❌");
            }
            
            System.out.printf("Sample batch accuracy: %d/10 (%.1f%%)\n", 
                correctInSample, correctInSample * 10.0);
        }
        
        System.out.println("\n💾 Saving EDUX CNN model...");
        saveModel(cnn, modelPath);
        
        System.out.println("\n🔬 Testing model persistence...");
        NeuralNetwork loadedModel = loadModel(modelPath);
        if (loadedModel != null) {
            System.out.println("✅ EDUX CNN model saved and loaded successfully!");
            
            Predictor loadedPredictor = new Predictor(loadedModel);
            testLoader.reset();
            if (testLoader.hasNext()) {
                var batch = testLoader.next();
                int[] loadedPredictions = loadedPredictor.predictBatch(batch.getData());
                System.out.printf("🔍 Loaded EDUX model made %d predictions successfully\n", 
                    loadedPredictions.length);
            }
            
            loadedModel.cleanup();
        } else {
            System.out.println("❌ Model save/load verification failed!");
        }
        
        cnn.cleanup();
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🎊 EDUX CNN Training Complete!");
        System.out.println("📁 Model saved as: " + modelPath);
        System.out.printf("⏱️  Total training time: %.1f minutes\n", totalTrainingTime / 60.0);
        System.out.printf("🎯 Final accuracy: %.2f%%\n", accuracy * 100);
        System.out.println("🚀 EDUX CNN ready for production use!");
        System.out.println("=".repeat(60));
    }
    
    private static void saveModel(NeuralNetwork model, String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(model);
            System.out.println("✅ EDUX model saved to: " + path);
        } catch (IOException e) {
            System.err.println("❌ Failed to save model: " + e.getMessage());
        }
    }
    
    private static NeuralNetwork loadModel(String path) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            NeuralNetwork model = (NeuralNetwork) ois.readObject();
            System.out.println("✅ EDUX model loaded from: " + path);
            return model;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("❌ Failed to load model: " + e.getMessage());
            return null;
        }
    }
}