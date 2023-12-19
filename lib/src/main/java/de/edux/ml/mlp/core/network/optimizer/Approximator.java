package de.edux.ml.mlp.core.network.optimizer;

import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.function.Function;

public class Approximator {
    public static Matrix gradient(Matrix input, Function<Matrix, Matrix> transofrm) {

        final double INC = 0.00000001;

        Matrix loss1 = transofrm.apply(input);

        if (loss1.getCols() != input.getCols()){
            throw new IllegalArgumentException("Input/Loss cols must be equal");
        }
        if (loss1.getRows() != 1){
            throw new IllegalArgumentException("Layer must return a row vector");
        }

        Matrix result = new Matrix(input.getRows(), input.getCols());

        input.forEach((row, col, index, value) -> {
            Matrix incremeted = input.addIncrement(row, col, INC);
            Matrix loss2 = transofrm.apply(incremeted);

            double rate = (loss2.getData()[col] - loss1.getData()[col]) / INC;
            result.set(row, col, rate);
        });

        return result;

    }

    public static Matrix weightGradient(Matrix weights, Function<Matrix, Matrix> transofrm) {

        final double INC = 0.00000001;

        Matrix loss1 = transofrm.apply(weights);
        Matrix result = new Matrix(weights.getRows(), weights.getCols(), (i) -> 0);

        weights.forEach((row, col, index, value) -> {
            Matrix incremeted = weights.addIncrement(row, col, INC);
            Matrix loss2 = transofrm.apply(incremeted);

            double rate = (loss2.get(0)- loss1.get(0)) / INC;
            result.set(row, col, rate);
        });

        return result;

    }
}
