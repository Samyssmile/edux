package de.edux.ml.knn;

public interface ILabeledPoint {
    double[] getFeatures();
    String getLabel();
}