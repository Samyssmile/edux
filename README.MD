[![Build](https://github.com/Samyssmile/edux/actions/workflows/gradle.yml/badge.svg?branch=main)](https://github.com/Samyssmile/edux/actions/workflows/gradle.yml)
[![CodeQL](https://github.com/Samyssmile/edux/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/Samyssmile/edux/actions/workflows/codeql-analysis.yml)

# EDUX - Java Machine Learning Library

EDUX is a user-friendly library for solving problems with a machine learning approach.

## Features

EDUX supports a variety of machine learning algorithms including:

- **Multilayer Perceptron (Neural Network):** Suitable for regression and classification problems, MLPs can approximate
  non-linear functions.
- **K Nearest Neighbors:** A simple, instance-based learning algorithm used for classification and regression.
- **Decision Tree:** Offers visual and explicitly laid out decision making based on input features.
- **Support Vector Machine:** Effective for binary classification, and can be adapted for multi-class problems.
- **RandomForest:** An ensemble method providing high accuracy through building multiple decision trees.

### Augmentations

Edux supports a variety of image augmentations, which can be used to increase the performance of your model.

### Few examples:

#### Color Equalization

<figure>
  <img src="https://github.com/Samyssmile/edux/assets/6922428/01d5a67c-0a62-4884-a2cc-7b0be1ee4601" width="300" alt="Original Image">
</figure>

<figure>
  <img src="https://github.com/Samyssmile/edux/assets/6922428/a3b04e8a-85c7-4bf3-8f76-9f8ce330e304" width="300" alt="Color Equalized Image">
</figure>

#### Monochrome + Noise

<figure>
  <img src="https://github.com/Samyssmile/edux/assets/6922428/56c4f7a4-93dc-483c-b5da-c8a15989b313" width="300" alt="Original Image">
</figure>

<figure>
  <img src="https://github.com/Samyssmile/edux/assets/6922428/25a8b2e5-0373-4781-8001-114e699fc2fe" width="300" alt="Monochrome + Noise Image">
</figure>

#### Code Example

#### Single Image

```
    AugmentationSequence augmentationSequence=
        new AugmentationBuilder()
        .addAugmentation(new ResizeAugmentation(250,250))
        .addAugmentation(new ColorEqualizationAugmentation())
        .build();

        BufferedImage augmentedImage=augmentationSequence.applyTo(image);
```

#### Run for all images in a directory

```
    AugmentationSequence augmentationSequence=
        new AugmentationBuilder()
        .addAugmentation(new ResizeAugmentation(250,250))
        .addAugmentation(new ColorEqualizationAugmentation())
        .addAugmentation(new BlurAugmentation(25))
        .addAugmentation(new RandomDeleteAugmentation(10,20,20))
        .build()
        .run(trainImagesDir,numberOfWorkers,outputDir);
```

### Battle Royale - Which algorithm is the best?

We run all algorithms on the same dataset and compare the results.
[Benchmark](https://github.com/Samyssmile/edux/discussions/42)

## Goal

The main goal of this project is to create a user-friendly library for solving problems using a machine learning
approach. The library is designed to be easy to use, enabling the solution of problems with just a few lines of code.

## Features

The library currently supports:

- Multilayer Perceptron (Neural Network)
- K Nearest Neighbors
- Decision Tree
- Support Vector Machine
- RandomForest

## Get started

Include the library as a dependency in your Java project file.

### Gradle

```
 implementation 'io.github.samyssmile:edux:1.0.7'
```

### Maven

```
  <dependency>
     <groupId>io.github.samyssmile</groupId>
     <artifactId>edux</artifactId>
     <version>1.0.7</version>
 </dependency>
```

### Hardware Acceleration (preview feature)

EDUX supports Nvidia GPU acceleration.

#### Requirements

- Nvidia GPU with CUDA support
- CUDA Toolkit 11.8

## Getting started tutorial

This section guides you through using EDUX to process your dataset, configure a multilayer perceptron (Multilayer Neural
Network), perform training and evaluation.

A multi-layer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of
input features. An MLP is characterized by several layers of input nodes connected as a directed graph between the input
and output layers.

### Step 0: Get Familiar with the Dataset

In this example we use the famouse MNIST Dataset. The MNIST database contains 60,000 training images and 10,000 testing

![](https://hc-linux.eu/edux/mnist-examples.png)

### Step 1: Data Processing

```
    String trainImages = "train-images.idx3-ubyte";
    String trainLabels = "train-labels.idx1-ubyte";
    String testImages = "t10k-images.idx3-ubyte";
    String testLabels = "t10k-labels.idx1-ubyte";
    Loader trainLoader = new ImageLoader(trainImages, trainLabels, batchSize);
    Loader testLoader = new ImageLoader(testImages, testLabels, batchSize);

```

### Step 2: Configure the MultilayerPerceptron

```
    int batchSize = 100;
    int threads = 1;
    int epochs = 10;
    float initialLearningRate = 0.1f;
    float finalLearningRate = 0.001f;

    MetaData trainMetaData = trainLoader.open();
    int inputSize = trainMetaData.getInputSize();
    int outputSize = trainMetaData.getExpectedSize();
    trainLoader.close();
```

### Step 3: Build the Network

We use the NetworkBuilder Class

```
    new NetworkBuilder()
        .addLayer(new DenseLayer(inputSize, 32))  //32 Neurons as output size
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(32, outputSize)) //32 Neurons as input size
        .addLayer(new SoftmaxLayer())
        .withBatchSize(batchSize)
        .withLearningRates(initialLearningRate, finalLearningRate)
        .withExecutionMode(singleThread)
        .withEpochs(epochs)
        .build()
        .printArchitecture()
        .fit(trainLoader, testLoader)
        .saveModel("model.edux"); // Save the trained model
```

### Step 4: Load the model and continue training

Load 'model.edux' and continue training for 10 epochs.

```
    NeuralNetwork nn =
        new NetworkBuilder().withEpochs(10).loadModel("model.edux").fit(trainLoader, testLoader);
```

### Results

```output
........................Epoch: 1, Loss: 1,14, Accuracy: 91,04
...
........................Epoch: 10, Loss: 0,13, Accuracy: 96,16
```

### Working examples

You can find more fully working examples for all algorithms in
the [examples](https://github.com/Samyssmile/edux/tree/main/example/src/main/java/de/example) folder.

For examples we use the

* [IRIS dataset](https://archive.ics.uci.edu/ml/datasets/iris).
* [SEABORNE PENGUINS dataset](https://seaborn.pydata.org/archive/0.11/tutorial/function_overview.html).

## Contributions

Contributions are warmly welcomed! If you find a bug, please create an issue with a detailed description of the problem.
If you wish to suggest an improvement or fix a bug, please make a pull request. Also checkout
the [Rules and Guidelines](https://github.com/Samyssmile/edux/wiki/Rules-&-Guidelines-for-New-Developers) page for more
information.
