package de.edux.ml.svm;

public interface ISupportVectorMachine {

    void train(double[][] features, int[] labels);

    // Methode zum Klassifizieren eines einzelnen Datenpunkts
    int predict(double[] features);

    // Methode zum Evaluieren der Leistung der SVM auf einem Testdatensatz
    double evaluate(double[][] features, int[] labels);
}
