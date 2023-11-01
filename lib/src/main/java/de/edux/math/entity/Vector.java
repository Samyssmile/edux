package de.edux.math.entity;

import de.edux.math.Entity;
import de.edux.math.Validations;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class Vector implements Entity<Vector>, Iterable<Double> {

  private final double[] raw;

  public Vector(double[] vector) {
    this.raw = vector;
  }

  @Override
  public Vector add(Vector another) {
    return add(another.raw());
  }

  public Vector add(double[] another) {
    Validations.size(raw, another);

    double[] result = new double[length()];
    for (int i = 0; i < result.length; i++) {
      result[i] = raw[i] + another[i];
    }

    return new Vector(result);
  }

  @Override
  public Vector subtract(Vector another) {
    return subtract(another.raw());
  }

  public Vector subtract(double[] another) {
    Validations.size(raw, another);

    double[] result = new double[length()];
    for (int i = 0; i < result.length; i++) {
      result[i] = raw[i] - another[i];
    }

    return new Vector(result);
  }

  @Override
  public Vector multiply(Vector another) {
    return multiply(another.raw());
  }

  public Vector multiply(double[] another) {
    Validations.size(raw, another);

    double[] result = new double[length()];
    for (int i = 0; i < result.length; i++) {
      result[i] = raw[i] * another[i];
      if (result[i] == 0) { // Avoiding -0 result
        result[i] = 0;
      }
    }

    return new Vector(result);
  }

  @Override
  public Vector scalarMultiply(double n) {
    double[] result = new double[length()];
    for (int i = 0; i < result.length; i++) {
      result[i] = raw[i] * n;
      if (result[i] == 0) { // Avoiding -0 result
        result[i] = 0;
      }
    }

    return new Vector(result);
  }

  public double dot(Vector another) {
    return dot(another.raw());
  }

  public double dot(double[] another) {
    Validations.size(raw, another);

    double result = 0;
    for (int i = 0; i < raw.length; i++) {
      result += raw[i] * another[i];
    }

    return result;
  }

  public int length() {
    return raw.length;
  }

  public double[] raw() {
    return raw.clone();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof Vector) {
      return Arrays.equals(raw, ((Vector) obj).raw());
    }
    return false;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder("[");
    for (int i = 0; i < raw.length; i++) {
      builder.append(raw[i]);
      if (i != raw.length - 1) {
        builder.append(", ");
      }
    }
    return builder.append("]").toString();
  }

  @Override
  public Iterator<Double> iterator() {
    return new VectorIterator(raw);
  }

  public static class VectorIterator implements Iterator<Double> {

    private final double[] data;
    private int current;

    public VectorIterator(double[] data) {
      this.data = data;
      this.current = 0;
    }

    @Override
    public boolean hasNext() {
      return current < data.length;
    }

    @Override
    public Double next() {
      if (!hasNext()) throw new NoSuchElementException();
      return data[current++];
    }
  }
}
