package de.edux.math;

public final class Validations {

  public static void size(double[] first, double[] second) {
    if (first.length != second.length) {
      throw new IllegalArgumentException("sizes mismatch");
    }
  }

  public static void sizeMatrix(double[][] first, double[][] second) {
    if (first.length != second.length || first[0].length != second[0].length) {
      throw new IllegalArgumentException("sizes mismatch");
    }
  }
}
