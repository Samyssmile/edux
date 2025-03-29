[![Build](https://github.com/Samyssmile/ejdux/actions/workflows/gradle.yml/badge.svg?branch=main)](https://github.com/Samyssmile/edux/actions/workflows/gradle.yml)
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

This section guides you through using EDUX to process your dataset, by configuring a decision tree 


A decision tree is a graphical representation of different options for solving a problem and shows how different factors are related. It has a hierarchical tree structure which starts with one main question at the top called a node which further branches out into different possible outcomes where:

* Root Node is the starting point that represents the entire dataset.

* Branches: These are the lines that connect nodes. It shows the flow from one decision to another.

* Internal Nodes are Points where decisions are made based on the input features.

* Leaf Nodes: These are the terminal nodes at the end of branches that represent final outcomes or predictions
Decision-Tree

![decision_tree.png](decision_tree.png)

### Step 0: Get Familiar with the Dataset

In this example we will use a dataset of seaborn penguins based on their
species, island, Bill length(mm), Bill depth(mm), flipper length (mm), Body mass (g) and their sex



### Step 1: Initiate 

```
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.70;
    private static final File CSV_FILE =
            new File("path-to-the-penguins.csv");
    private static final boolean SKIP_HEAD = true;
    private static final ImputationStrategy averageImputation = ImputationStrategy.AVERAGE;
    private static final ImputationStrategy modeImputation = ImputationStrategy.MODE;

```

### Step 2: Load the csv file and process the data

```
    var featureColumnIndices = new int[] {1, 2, 3, 4, 5, 6};
    var targetColumnIndex = 0;

    var penguinsDataProcessor =
        new DataProcessor(new CSVIDataReader())
            .loadDataSetFromCSV(CSV_FILE, ',', SKIP_HEAD, featureColumnIndices, targetColumnIndex)
            .imputation(0, modeImputation)
            .imputation(1, modeImputation)
            .imputation(2, averageImputation)
            .imputation(3, averageImputation)
            .imputation(4, averageImputation)
            .imputation(5, averageImputation)
            .imputation(6, modeImputation)
            .normalize()
            .shuffle()
            .split(TRAIN_TEST_SPLIT_RATIO);
```

### Step 3: Enable the classifier

```
    Classifier classifier = new DecisionTree(2, 2, 3, 12);

    var trainFeatures = penguinsDataProcessor.getTrainFeatures(featureColumnIndices);
    var trainTestFeatures = penguinsDataProcessor.getTestFeatures(featureColumnIndices);
    var trainLabels = penguinsDataProcessor.getTrainLabels(targetColumnIndex);
    var trainTestLabels = penguinsDataProcessor.getTestLabels(targetColumnIndex);

    classifier.train(trainFeatures, trainLabels);
    classifier.evaluate(trainTestFeatures, trainTestLabels);
```

### Results

```output
23:02:19.211 [main] INFO  de.edux.data.provider.DataProcessor - Dataset loaded
23:02:20.341 [main] INFO  de.edux.ml.decisiontree.DecisionTree - Decision Tree -  accuracy: 94.23%

Process finished with exit code 0
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
