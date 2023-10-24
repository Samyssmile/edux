package de.edux.math;

/**
 * @author ImGolem?
 */
public final class MathUtil {

    public static double[] unwrap(double[][] matrix) {
        double[] result = new double[matrix.length * matrix[0].length];
        int i = 0;
        for (double[] arr : matrix) {
            for (double val : arr) {
                result[i++] = val;
            }
        }
        return result;
    }

}
