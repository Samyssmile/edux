package de.edux.math.entity;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class MatrixTest {

  static Matrix first;
  static Matrix second;

  @BeforeEach
  public void init() {
    first =
        new Matrix(
            new double[][] {
              {5, 3, -1},
              {-2, 0, 6},
              {5, 1, -9}
            });
    second =
        new Matrix(
            new double[][] {
              {8, 7, 4},
              {1, -5, 2},
              {0, 3, 0}
            });
  }

  @Test
  public void testAdd() {
    assertEquals(
        new Matrix(
            new double[][] {
              {13, 10, 3},
              {-1, -5, 8},
              {5, 4, -9}
            }),
        first.add(second));
  }

  @Test
  public void testSubtract() {
    assertEquals(
        new Matrix(
            new double[][] {
              {-3, -4, -5},
              {-3, 5, 4},
              {5, -2, -9}
            }),
        first.subtract(second));
  }

  @Test
  public void testScalarMultiply() {
    assertEquals(
        new Matrix(
            new double[][] {
              {20, 12, -4},
              {-8, 0, 24},
              {20, 4, -36}
            }),
        first.scalarMultiply(4));
    assertEquals(
        new Matrix(
            new double[][] {
              {-48, -42, -24},
              {-6, 30, -12},
              {0, -18, 0}
            }),
        second.scalarMultiply(-6));
  }
}
