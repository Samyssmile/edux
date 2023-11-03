package de.edux.core.math.matrix.strassen;

import de.edux.core.math.IMatrixArithmetic;
import java.util.concurrent.ForkJoinPool;

public class StrassenParallelInplace implements IMatrixArithmetic {
  private ForkJoinPool forkJoinPool = new ForkJoinPool();

  @Override
  public double[][] multiply(double[][] matrixA, double[][] matrixB) {}

  private int nextPowerOfTwo(int number) {
    int power = 1;
    while (power < number) {
      power *= 2;
    }
    return power;
  }
}
