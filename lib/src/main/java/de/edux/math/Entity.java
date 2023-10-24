package de.edux.math;

/**
 * @author ImGolem?
 */
public interface Entity<T> {

    T add(T another);

    T subtract(T another);

    T multiply(T another);

    T scalarMultiply(double n);

}
