package de.nexent.edux.ml.nn.config;

import de.nexent.edux.functions.activation.ActivationFunction;
import de.nexent.edux.functions.loss.LossFunction;

import java.util.List;

public record Configuration(int inputSize, List<Integer> hiddenLayersSize, int outputSize, double learningRate, int epochs,
                            ActivationFunction hiddenLayerActivationFunction, ActivationFunction outputLayerActivationFunction, LossFunction lossFunction) {

}