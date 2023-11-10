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

````java
    AugmentationSequence augmentationSequence=
        new AugmentationBuilder()
        .addAugmentation(new ResizeAugmentation(250,250))
        .addAugmentation(new ColorEqualizationAugmentation())
        .build();

        BufferedImage augmentedImage=augmentationSequence.applyTo(image);
````

#### Run for all images in a directory

```java
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
 implementation 'io.github.samyssmile:edux:1.0.6'
```

### Maven

```
  <dependency>
     <groupId>io.github.samyssmile</groupId>
     <artifactId>edux</artifactId>
     <version>1.0.6</version>
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

![Neural Network](https://hc-linux.eu/github/iris-nn.png)

### Step 1: Data Processing

Firstly, we will load and prepare the IRIS dataset:

| sepal.length | sepal.width | petal.length | petal.width | variety |
|--------------|-------------|--------------|-------------|---------|
| 5.1          | 3.5         | 1.4          | 0.2         | Setosa  |

```java
var featureColumnIndices=new int[]{0,1,2,3}; // Specify your feature columns
        var targetColumnIndex=4; // Specify your target column

        var dataProcessor=new DataProcessor(new CSVIDataReader());
        var dataset=dataProcessor.loadDataSetFromCSV(
        "path/to/your/data.csv", // Replace with your CSV file path
        ',',                     // CSV delimiter
        true,                    // Whether to skip the header
        featureColumnIndices,
        targetColumnIndex
        );
        dataset.shuffle();
        dataset.normalize();
        dataProcessor.split(0.8); // Replace with your train-test split ratio
```

### Step 2: Preparing Training and Test Sets:

Extract the features and labels for both training and test sets:

```java
    var trainFeatures=dataProcessor.getTrainFeatures(featureColumnIndices);
        var trainLabels=dataProcessor.getTrainLabels(targetColumnIndex);
        var testFeatures=dataProcessor.getTestFeatures(featureColumnIndices);
        var testLabels=dataProcessor.getTestLabels(targetColumnIndex);
```

### Step 3: Network Configuration

```java
var networkConfiguration=new NetworkConfiguration(
        trainFeatures[0].length,     // Number of input neurons
        List.of(128,256,512),      // Number of neurons in each hidden layer
        3,                           // Number of output neurons
        0.01,                        // Learning rate
        300,                         // Number of epochs
        ActivationFunction.LEAKY_RELU, // Activation function for hidden layers
        ActivationFunction.SOFTMAX,    // Activation function for output layer
        LossFunction.CATEGORICAL_CROSS_ENTROPY, // Loss function
        Initialization.XAVIER,        // Weight initialization for hidden layers
        Initialization.XAVIER         // Weight initialization for output layer
        );
```

### Step 4: Training and Evaluation

```java
MultilayerPerceptron multilayerPerceptron=new MultilayerPerceptron(
        networkConfiguration,
        testFeatures,
        testLabels
        );
        multilayerPerceptron.train(trainFeatures,trainLabels);
        multilayerPerceptron.evaluate(testFeatures,testLabels);
```

### Results

```output
...
MultilayerPerceptron - Best accuracy after restoring best MLP model: 98,56%
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
