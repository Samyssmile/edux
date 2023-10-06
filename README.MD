[![Build](https://github.com/Samyssmile/edux/actions/workflows/gradle.yml/badge.svg?branch=main)](https://github.com/Samyssmile/edux/actions/workflows/gradle.yml)
[![CodeQL](https://github.com/Samyssmile/edux/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/Samyssmile/edux/actions/workflows/codeql-analysis.yml)
# EDUX - Java Machine Learning Library

EDUX is a user-friendly library for solving problems with a machine learning approach.

## Features

EDUX supports a variety of machine learning algorithms including:

- **Multilayer Perceptron (Neural Network):** Suitable for regression and classification problems, MLPs can approximate non-linear functions.
- **K Nearest Neighbors:** A simple, instance-based learning algorithm used for classification and regression.
- **Decision Tree:** Offers visual and explicitly laid out decision making based on input features.
- **Support Vector Machine:** Effective for binary classification, and can be adapted for multi-class problems.
- **RandomForest:** An ensemble method providing high accuracy through building multiple decision trees.


## Goal
The main goal of this project is to create a user-friendly library for solving problems using a machine learning approach. The library is designed to be easy to use, enabling the solution of problems with just a few lines of code.

## Features

The library currently supports:

- Multilayer Perceptron (Neural Network)
- K Nearest Neighbors
- Decision Tree
- Support Vector Machine
- RandomForest

## Installation

Include the library as a dependency in your Java project file.

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

## How to use this library

        NetworkConfiguration networkConfiguration = new NetworkConfiguration(...ActivationFunction.LEAKY_RELU, ActivationFunction.SOFTMAX, LossFunction.CATEGORICAL_CROSS_ENTROPY, Initialization.XAVIER, Initialization.XAVIER);
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron(features, labels, testFeatures, testLabels, networkConfiguration);
        multilayerPerceptron.train();

        multilayerPerceptron.predict(...);

### Working examples
You can find working examples for all algorithms in the [examples](https://github.com/Samyssmile/edux/tree/main/example/src/main/java/de/example) folder.

In all examples the IRIS or Seaborn Pinguins datasets are used.

#### Iris Dataset
The IRIS dataset is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in his 1936 paper *The use of multiple measurements in taxonomic problems*

![Neural Network](https://hc-linux.eu/github/iris-nn.png)
## Contributions


Contributions are warmly welcomed! If you find a bug, please create an issue with a detailed description of the problem. If you wish to suggest an improvement or fix a bug, please make a pull request. Also checkout the [Rules and Guidelines](https://github.com/Samyssmile/edux/wiki/Rules-&-Guidelines-for-New-Developers) page for more information.
