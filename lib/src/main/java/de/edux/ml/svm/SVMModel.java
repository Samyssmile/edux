package de.edux.ml.svm;

public class SVMModel {

    private final SVMKernel kernel;
    private double c;
    private double[] weights;
    private double bias = 0.0;

    public SVMModel(SVMKernel kernel, double c) {
        this.kernel = kernel;
        this.c = c;
    }

    public void train(double[][] features, int[] labels) {
        int n = features[0].length;
        weights = new double[n];
        int iterations = 10000;

        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < features.length; i++) {
                double[] xi = features[i];
                int target = labels[i];
                double prediction = predict(xi);

                if (target * prediction < 1) {
                    for (int j = 0; j < n; j++) {
                        weights[j] = weights[j] + c * (target * xi[j] - 2 * (1/iterations) * weights[j]);
                    }
                    bias += c * target;
                } else {
                    for (int j = 0; j < n; j++) {
                        weights[j] = weights[j] - c * 2 * (1/iterations) * weights[j];
                    }
                }
            }
        }
    }

    public int predict(double[] features) {
        double result = bias;
        for (int i = 0; i < weights.length; i++) {
            result += weights[i] * features[i];
        }
        return (result >= 1) ? 1 : -1;
    }
}
