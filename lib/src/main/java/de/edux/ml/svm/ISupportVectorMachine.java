package de.edux.ml.svm;

public interface ISupportVectorMachine {

    void train(double[][] features, int[] labels);

    int predict(double[] features);

    double evaluate(double[][] features, int[] labels);
}
