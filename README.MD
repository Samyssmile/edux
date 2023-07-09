# EDUX Neural Network Library

EDUX Neural Network Library is a user-friendly library for implementing machine learning algorithms and artificial neural networks in Java. It offers a straightforward and flexible interface to construct, train, and utilize neural networks.

1. [Goal](#goal)
2. [Installation](#installation)
3. [Usage - Neural Network](#usage---neural-network)
4. [K Nearest Neighbors](#k-nearest-neighbors)
5. [Features](#features)
6. [Contributions](#contributions)



## Goal 
Main goal of this project is a educational purpose.

## Installation

To install, include the library as a dependency in your Java project file. As the library is not yet available on a package management repository, you will have to manually download the repository and include it in your project.

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

    package de.edux.example.knn.KnnMain.java

1. Implement IDataProvider Interface to get your data.
2. Transform your data to LabeledPoints.
3. Create a KnnClassifier with the number of neighbors and the labeled points.

You can evaluate the accuracy of the classifier with the evaluate method.

    knnClassifier.evaluate(testLabeledPoints);
    
    output
    [main] INFO  de.edux.ml.knn.KnnClassifier - Accuracy: 98.33333333333333%

To classify a single record use the classify method.

    knnClassifier.classify(...);

## Features

The library currently supports:

- Multilayer Perceptron
- K Nearest Neighbors
- Descision Tree (progress)
- Support Vector Machine (progress)


![Neural Network](https://hc-linux.eu/github/iris-nn.png)
## Contributions

Contributions are warmly welcomed! If you find a bug, please create an issue with a detailed description of the problem. If you wish to suggest an improvement or fix a bug, please make a pull request.