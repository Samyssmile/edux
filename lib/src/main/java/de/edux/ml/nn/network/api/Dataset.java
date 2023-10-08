package de.edux.ml.nn.network.api;

import java.util.List;

public record Dataset<T>(List<T> trainData, List<T> testData) {}
