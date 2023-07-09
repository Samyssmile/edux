package de.edux.functions.loss;

public enum LossFunction {

    CATEGORICAL_CROSS_ENTROPY {
        @Override
        public double calculateError(double[] output, double[] target) {
            double error = 0;
            for (int i = 0; i < target.length; i++) {
                error += target[i] * Math.log(output[i]);
            }
            return -error;
        }
    },
    MEAN_SQUARED_ERROR {
        @Override
        public double calculateError(double[] output, double[] target) {
            double error = 0;
            for (int i = 0; i < target.length; i++) {
                error += Math.pow(target[i] - output[i], 2);
            }
            return error / target.length;
        }
    },
    MEAN_ABSOLUTE_ERROR {
        @Override
        public double calculateError(double[] output, double[] target) {
            double error = 0;
            for (int i = 0; i < target.length; i++) {
                error += Math.abs(target[i] - output[i]);
            }
            return error / target.length;
        }
    },
    HINGE_LOSS {
        @Override
        public double calculateError(double[] output, double[] target) {
            double error = 0;
            for (int i = 0; i < target.length; i++) {
                error += Math.max(0, 1 - target[i] * output[i]);
            }
            return error / target.length;
        }
    },
    SQUARED_HINGE_LOSS {
        @Override
        public double calculateError(double[] output, double[] target) {
            double error = 0;
            for (int i = 0; i < target.length; i++) {
                error += Math.pow(Math.max(0, 1 - target[i] * output[i]), 2);
            }
            return error / target.length;
        }
    },
    BINARY_CROSS_ENTROPY {
        @Override
        public double calculateError(double[] output, double[] target) {
            double error = 0;
            for (int i = 0; i < target.length; i++) {
                error += target[i] * Math.log(output[i]) + (1 - target[i]) * Math.log(1 - output[i]);
            }
            return -error;
        }
    };

    public abstract double calculateError(double[] target, double[] output);
}

