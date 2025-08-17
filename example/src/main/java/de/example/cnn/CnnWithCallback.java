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
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *  Download Dataset from https://github.com/rasbt/mnist-pngs and put it into edux/example/datasets/mnist-pngs
 *
 *  Showcases a small CNN model with callbacks for monitoring training progress.
 *  This example uses a smaller architecture suitable for quick training and testing.
 */
public class CnnWithCallback {
	public static void main(String[] args) {
		System.out.println("Starting Small MNIST CNN Training with EDUX Library");
		System.out.println("==================================================");


		Path datasetRoot = Paths.get("example", "datasets", "mnist-pngs");
		Path trainCsv    = datasetRoot.resolve("train.csv");
		Path testCsv     = datasetRoot.resolve("test.csv");
		Path modelPath   = Paths.get("edux-small-cnn-model.ser");

		int    batchSize    = 256;
		int    epochs       = 2;
		double learningRate = 0.01;


		System.out.println("Loading dataset...");
		MNISTDataLoader trainLoader = new MNISTDataLoader(trainCsv.toString(), datasetRoot.toString(), batchSize, 0.20);
		MNISTDataLoader testLoader  = new MNISTDataLoader(testCsv.toString(), datasetRoot.toString(), batchSize, 1);


		NeuralNetwork smallCnn = new NetworkBuilder()
				.addLayer(new ConvolutionalLayer(1, 8, 5, 1, 2)) // Weniger Filter, größerer Kernel
				.addLayer(new ReLuLayer())
				.addLayer(new PoolingLayer(PoolingLayer.PoolingType.MAX, 2))
				.addLayer(new FlattenLayer())
				.addLayer(new FullyConnectedLayer(8 * 14 * 14, 64))
				.addLayer(new ReLuLayer())
				.addLayer(new FullyConnectedLayer(64, 10))
				.build();


		System.out.println("Setting up training...");
		CrossEntropyLoss lossFunction = new CrossEntropyLoss();
		SGD              optimizer    = new SGD(learningRate);
		Trainer          trainer      = new Trainer(smallCnn, lossFunction, optimizer);

		trainer.addCallback(new Callback() {
			private long   epochStartTime;
			private double bestLoss = Double.MAX_VALUE;

			@Override
			public void onEpochStart(TrainerContext ctx) {
				epochStartTime = System.currentTimeMillis();
				System.out.printf("\n🔄 Starting Epoch %d/%d with Small EDUX CNN\n", ctx.getCurrentEpoch() + 1, epochs);
				System.out.println("─────────────────────────────────────────");
			}

			@Override
			public void onEpochEnd(TrainerContext ctx) {
				long   epochTime   = System.currentTimeMillis() - epochStartTime;
				double currentLoss = ctx.getCurrentLoss();

				System.out.printf("\n✅ Epoch %d completed in %.1f seconds\n",
				                  ctx.getCurrentEpoch() + 1, epochTime / 1000.0);
				System.out.printf("   Final Loss: %.6f", currentLoss);

				if (currentLoss < bestLoss) {
					bestLoss = currentLoss;
					System.out.print(" 🌟 (New Best!)");
				}
				System.out.println();

				if (currentLoss < 0.2) {
					System.out.println("   📈 Good convergence for small model!");
				} else if (currentLoss < 0.5) {
					System.out.println("   📊 Making progress...");
				} else if (currentLoss < 1.0) {
					System.out.println("   📉 Learning in progress...");
				}
			}

			@Override
			public void onBatchEnd(TrainerContext ctx) {
				if (ctx.getCurrentBatch() % 100 == 0) {
					System.out.printf("   Batch %3d - Loss: %.4f\n",
					                  ctx.getCurrentBatch(), ctx.getCurrentLoss());
				}
			}
		});

		System.out.println("🚀 Starting training with Small EDUX CNN...");

		trainLoader.shuffle();
		long trainingStartTime = System.currentTimeMillis();
		trainer.train(trainLoader, epochs);
		long trainingEndTime = System.currentTimeMillis();

		double totalTrainingTime = (trainingEndTime - trainingStartTime) / 1000.0;
		System.out.printf("\n🎉 Small model training completed in %.1f seconds (%.1f minutes)\n",
		                  totalTrainingTime, totalTrainingTime / 60.0);

		System.out.println("\n📊 Evaluating Small EDUX CNN model...");
		float accuracy = trainer.evaluate(testLoader);
		System.out.printf("🎯 Test Accuracy: %.2f%%\n", accuracy * 100);

		if (accuracy > 0.95) {
			System.out.println("🏆 Excellent performance for a small model!");
		} else if (accuracy > 0.90) {
			System.out.println("🥇 Very good performance for a small model!");
		} else if (accuracy > 0.85) {
			System.out.println("🥈 Good performance for a small model!");
		} else {
			System.out.println("🥉 Reasonable performance. Small models have limitations.");
		}

		System.out.println("\n🔍 Making sample predictions with Small EDUX CNN...");
		Predictor predictor = new Predictor(smallCnn);

		testLoader.reset();
		if (testLoader.hasNext()) {
			var   batch       = testLoader.next();
			int[] predictions = predictor.predictBatch(batch.getData());

			FloatTensor labels        = (FloatTensor) batch.getLabels();
			float[]     labelData     = labels.getPrimitiveData();
			int         testBatchSize = labels.getShape()[0];
			int         numClasses    = labels.getShape()[1];

			System.out.println("Sample predictions vs actual labels:");
			int correctInSample = 0;
			for (int i = 0; i < Math.min(5, testBatchSize); i++) {
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

			System.out.printf("Sample batch accuracy: %d/5 (%.1f%%)\n",
			                  correctInSample, correctInSample * 20.0);
		}

		System.out.println("\n💾 Saving Small EDUX CNN model...");
		saveModel(smallCnn, modelPath.toString());

		System.out.println("\n🔬 Testing model persistence...");
		NeuralNetwork loadedModel = loadModel(modelPath.toString());
		if (loadedModel != null) {
			System.out.println("✅ Small EDUX CNN model saved and loaded successfully!");
			loadedModel.cleanup();
		} else {
			System.out.println("❌ Model save/load verification failed!");
		}

		smallCnn.cleanup();

		System.out.println("\n" + "=".repeat(50));
		System.out.println("🎊 Small EDUX CNN Training Complete!");
		System.out.println("📁 Model saved as: " + modelPath.toString());
		System.out.printf("⏱️  Total training time: %.1f minutes\n", totalTrainingTime / 60.0);
		System.out.printf("🎯 Final accuracy: %.2f%%\n", accuracy * 100);
		System.out.println("🚀 Small EDUX CNN ready for testing!");
		System.out.println("=".repeat(50));
	}

	private static void saveModel(NeuralNetwork model, String path) {
		try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
			oos.writeObject(model);
			System.out.println("✅ Small EDUX model saved to: " + path);
		} catch (IOException e) {
			System.err.println("❌ Failed to save model: " + e.getMessage());
		}
	}

	private static NeuralNetwork loadModel(String path) {
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
			NeuralNetwork model = (NeuralNetwork) ois.readObject();
			System.out.println("✅ Small EDUX model loaded from: " + path);
			return model;
		} catch (IOException | ClassNotFoundException e) {
			System.err.println("❌ Failed to load model: " + e.getMessage());
			return null;
		}
	}
}