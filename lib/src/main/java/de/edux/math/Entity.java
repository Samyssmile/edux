package de.edux.math;

public interface Entity<T> {

  T add(T another);

  T subtract(T another);

  T multiply(T another);

  T scalarMultiply(double n);
}
