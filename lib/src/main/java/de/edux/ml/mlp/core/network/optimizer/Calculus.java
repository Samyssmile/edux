package de.edux.ml.mlp.core.network.optimizer;

import java.util.function.DoubleFunction;

public class Calculus {
    private static final double INC = 1e-4;

    public static double func1(double x) {
        return 3.7 * x + 5.3;
    }

    public static double func2(double x) {
        return x * x;
    }

    public static double func3(double y1, double y2) {
        return y1 * y2 +4.7 *y1;
    }

    public static double differentiate(DoubleFunction<Double> function, double x) {
        double output1 = function.apply(x);
        double output2 = function.apply(x + INC);

        return (output2 - output1) / INC;
    }

    public static void main(String[] args) {
        double x = 3.64;
        double y = func1(x);
        double z = func2(y);

        double dydx = differentiate(Calculus::func1, x);
        double dzdy = differentiate(Calculus::func2, y);


    }

}
