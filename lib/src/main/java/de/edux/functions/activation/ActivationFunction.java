package de.edux.functions.activation;

public enum ActivationFunction {

    SIGMOID {
        @Override
        public double calculateActivation(double x) {
            return 1 / (1 + Math.exp(-x));
        }

        @Override
        public double calculateDerivative(double x) {
            return calculateActivation(x) * (1 - calculateActivation(x));
        }
    },
    RELU {
        @Override
        public double calculateActivation(double x) {
            return Math.max(0, x);
        }

        @Override
        public double calculateDerivative(double x) {
            return x > 0 ? 1 : 0;
        }
    },
    LEAKY_RELU {
        @Override
        public double calculateActivation(double x) {
            return Math.max(0.01 * x, x);
        }

        @Override
        public double calculateDerivative(double x) {
            if (x > 0) {
                return 1.0;
            } else {
                return 0.01;
            }
        }
    },
    TANH {
        @Override
        public double calculateActivation(double x) {
            return Math.tanh(x);
        }

        @Override
        public double calculateDerivative(double x) {
            return 1 - Math.pow(calculateActivation(x), 2);
        }
    }, SOFTMAX {
        @Override
        public double calculateActivation(double x) {
            return Math.exp(x);
        }

        @Override
        public double calculateDerivative(double x) {
            return calculateActivation(x) * (1 - calculateActivation(x));
        }

        @Override
        public double[] calculateActivation(double[] x) {
            double max = Double.NEGATIVE_INFINITY;
            for (double value : x)
                if (value > max)
                    max = value;

            double sum = 0.0;
            for (int i = 0; i < x.length; i++) {
                x[i] = Math.exp(x[i] - max);
                sum += x[i];
            }

            for (int i = 0; i < x.length; i++)
                x[i] /= sum;

            return x;
        }
    };


    public abstract double calculateActivation(double x);

    public abstract double calculateDerivative(double x);

    public  double[] calculateActivation(double[] x){
        throw new UnsupportedOperationException("Not implemented");
    }
}
