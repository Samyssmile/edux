package de.edux.ml.knn;

public class KnnPoint implements ILabeledPoint{

    private final String label;
    private final double[] features;

    public KnnPoint(double[] features, String label) {
        this.features = features;
        this.label = label;
    }

    @Override
    public double[] getFeatures() {
        return features;
    }

    @Override
    public String getLabel() {
        return label;
    }
}
