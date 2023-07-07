package de.edux.nn.config;

import de.edux.functions.activation.ActivationFunction;
import de.edux.functions.loss.LossFunction;

import java.util.List;

public record Configuration(int inputSize, List<Integer> hiddenLayersSize, int outputSize, double learningRate, int epochs,
                            ActivationFunction hiddenLayerActivationFunction, ActivationFunction outputLayerActivationFunction, LossFunction lossFunction) {

}