package de.edux.ml.nn.config;

import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.initialization.Initialization;
import de.edux.functions.loss.LossFunction;
import java.util.List;

public record NetworkConfiguration(
    int inputSize,
    List<Integer> hiddenLayersSize,
    int outputSize,
    double learningRate,
    int epochs,
    ActivationFunction hiddenLayerActivationFunction,
    ActivationFunction outputLayerActivationFunction,
    LossFunction lossFunction,
    Initialization hiddenLayerWeightInitialization,
    Initialization outputLayerWeightInitialization) {}
