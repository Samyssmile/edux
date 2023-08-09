package de.edux.functions.initialization;

public enum Initialization {
    //Xavier and HE
    XAVIER {
        @Override
        public double[] weightInitialization(int inputSize, double[] weights) {
            double xavier = Math.sqrt(6.0 / (inputSize + 1));
            for (int i = 0; i < weights.length; i++) {
                weights[i] = Math.random() * 2 * xavier - xavier;
            }
            return weights;
        }
    },
    HE {
        @Override
        public double[] weightInitialization(int inputSize, double[] weights) {
            double he = Math.sqrt(2.0 / inputSize);
            for (int i = 0; i < weights.length; i++) {
                weights[i] = Math.random() * 2 * he - he;
            }
            return weights;
        }
    };

    public abstract double[] weightInitialization(int inputSize, double[] weights);
}
