package de.example.data.iris;

public class Iris {
    public double sepalLength;
    public double sepalWidth;
    public double petalLength;
    public double petalWidth;
    public String variety;

    public Iris(double sepalLength, double sepalWidth, double petalLength, double petalWidth, String variety) {
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
        this.variety = variety;
    }

    @Override
    public String toString() {
        return "{sepalLength=" + sepalLength +
                ", sepalWidth=" + sepalWidth +
                ", petalLength=" + petalLength +
                ", petalWidth=" + petalWidth +
                ", variety='" + variety + '\'' +
                '}';
    }

    public double[] getFeatures() {
        return new double[]{sepalLength, sepalWidth, petalLength, petalWidth};
    }
}
