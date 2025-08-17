package de.example.cnn;

import de.edux.ml.cnn.data.MNISTDataLoader;
import de.edux.ml.cnn.layer.*;
import de.edux.ml.cnn.loss.CrossEntropyLoss;
import de.edux.ml.cnn.network.NetworkBuilder;
import de.edux.ml.cnn.network.NeuralNetwork;
import de.edux.ml.cnn.optimizer.SGD;
import de.edux.ml.cnn.training.Trainer;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *  Download Dataset from https://github.com/rasbt/mnist-pngs and put it into edux/example/datasets/mnist-pngs
 */
public class CnnMinimalExample {
	public static void main(String[] args) {
		System.out.println("Starting Small MNIST CNN Training with EDUX Library");
		System.out.println("==================================================");

		Path datasetRoot = Paths.get("example", "datasets", "mnist-pngs");
		Path trainCsv    = datasetRoot.resolve("train.csv");
		Path testCsv     = datasetRoot.resolve("test.csv");

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


		trainLoader.shuffle();
		trainer.train(trainLoader, epochs);
		float accuracy = trainer.evaluate(testLoader);

		System.out.println("Accuracy after " + epochs + " epochs: " + accuracy * 100 + "%");


	}

}