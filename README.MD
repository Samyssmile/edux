[![Build](https://github.com/Samyssmile/edux/actions/workflows/gradle.yml/badge.svg?branch=main)](https://github.com/Samyssmile/edux/actions/workflows/gradle.yml)
[![CodeQL](https://github.com/Samyssmile/edux/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/Samyssmile/edux/actions/workflows/codeql-analysis.yml)
# EDUX - Java Machine Learning Library

EDUX is a user-friendly library for solving problems with a machine learning approach.

1. [Goal](#goal)
2. [Installation](#installation)
3. [Usage - Neural Network](#usage---neural-network)
4. [K Nearest Neighbors](#k-nearest-neighbors)
5. [Features](#features)
6. [Contributions](#contributions)

## Goal 
The main goal of this project is to create a user-friendly library for solving problems using a machine learning approach. The library is designed to be easy to use, enabling the solution of problems with just a few lines of code. Please note that the library is currently under development and is not yet ready for production use.

## Installation

To install, include the library as a dependency in your Java project file.

### Gradle

```
 implementation 'io.github.samyssmile:edux:1.0.4'
```

### Maven
```
  <dependency>
     <groupId>io.github.samyssmile</groupId>
     <artifactId>edux</artifactId>
     <version>1.0.4</version>
 </dependency>
```

## Usage - Neural Network

1. You need to implement a IDataProvider Interface to provide the data for the network. Example for the IRIS-Dataset.
 ```
   public class IrisProvider implements IDataProvider<Iris>
 ```
2. Create a Neural Network Configuration.
```
    var neuralNetworkConfiguration = new Configuration(
                                         inputSize, 
                                         hiddenLayersSize, 
                                         outputSize, 
                                         learningRate,
                                         epochs,
                                         ActivationFunction.LEAKY_RELU, // Activation function of the hidden layers
                                         ActivationFunction.SOFTMAX, // Activation function of the output layer
                                         LossFunction.CATEGORICAL_CROSS_ENTROPY);// Loss function of the network
```
3. Create a Neural Network with the configuration and data from provider. Finaly train the network.
```
   var neuralNetwork = new NeuralNetwork(features, labels, testFeatures, testLabels, neuralNetworkConfiguration);
   neuralNetwork.train();
```

You can predict the output with the predict method.
```
   var prediction = neuralNetwork.predict(yourInputYouWantToPredict);
```



## K Nearest Neighbors

Its also possible to use the K Nearest Neighbors algorithm. 

    KnnClassifier knnClassifier = new KnnClassifier(1, labeledPoints);

a full working example can be found in 

    package knn.example.de.nexent.edux.KnnExample.java

1. Implement IDataProvider Interface to get your data.
2. Transform your data to LabeledPoints.
3. Create a KnnClassifier with the number of neighbors and the labeled points.

You can evaluate the accuracy of the classifier with the evaluate method.

    knnClassifier.evaluate(testLabeledPoints);
    
    output
    [main] INFO  knn.ml.de.nexent.edux.KnnClassifier - Accuracy: 98.33333333333333%

To classify a single record use the classify method.

    knnClassifier.classify(...);

## Decision Tree
A full working example can be found in 

    package decisiontree.example.de.nexent.edux.DecisionTreeExample.java

Example Output for IRIS Datenset

    de.edux.ml.decisiontree.DecisionTree - Accuracy: 95,0000%
    Feature Importance: [0.4444444444444444, 0.0, 0.4291223175466741, 0.42559077319203725]

## Features

The library currently supports:

- Multilayer Perceptron
- K Nearest Neighbors
- Descision Tree
- Support Vector Machine (progress)
- Random Forest (progress)


![Neural Network](https://hc-linux.eu/github/iris-nn.png)
## Contributions

Contributions are warmly welcomed! If you find a bug, please create an issue with a detailed description of the problem. If you wish to suggest an improvement or fix a bug, please make a pull request.
