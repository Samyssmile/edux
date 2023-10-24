package de.edux.math;

/**
 * @author ImGolem?
 */
public final class Validations {

    public static void size(double[] f, double[] s) {
        if (f.length != s.length) {
            throw new IllegalArgumentException("sizes mismatch");
        }
    }

    public static void sizeMatrix(double[][] f, double[][] s) {
        if (f.length != s.length || f[0].length != s[0].length) {
            throw new IllegalArgumentException("sizes mismatch");
        }
    }

}
