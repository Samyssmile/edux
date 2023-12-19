package de.edux.ml.mlp.util;

public class TrainingArrays {
    private double[] input;
    private double[] output;

    public TrainingArrays(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getOutput() {
        return output;
    }

    public void setOutput(double[] output) {
        this.output = output;
    }
}
