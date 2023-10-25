package de.edux.math.entity;

import de.edux.math.Entity;
import de.edux.math.MathUtil;
import de.edux.math.Validations;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Matrix implements Entity<Matrix>, Iterable<Double> {

    private final double[][] raw;

    public Matrix(double[][] matrix) {
        this.raw = matrix;
    }

    @Override
    public Matrix add(Matrix another) {
        return add(another.raw());
    }

    public Matrix add(double[][] another) {
        Validations.sizeMatrix(raw, another);

        double[][] result = new double[raw.length][raw[0].length];

        for (int i = 0; i < result.length; i++) {
            for (int a = 0; a < result[0].length; a++) {
                result[i][a] = raw[i][a] + another[i][a];
            }
        }

        return new Matrix(result);
    }

    @Override
    public Matrix subtract(Matrix another) {
        return subtract(another.raw());
    }

    @Override
    public Matrix multiply(Matrix another) {
        return multiply(another.raw());
    }

    public Matrix multiply(double[][] another) {
        return null; // TODO optimized algorithm for matrix multiplication
    }

    @Override
    public Matrix scalarMultiply(double n) {
        double[][] result = new double[raw.length][raw[0].length];

        for (int i = 0; i < result.length; i++) {
            for (int a = 0; a < result[0].length; a++) {
                result[i][a] = raw[i][a] * n;
            }
        }

        return new Matrix(result);
    }

    public Matrix subtract(double[][] another) {
        Validations.sizeMatrix(raw, another);

        double[][] result = new double[raw.length][raw[0].length];

        for (int i = 0; i < result.length; i++) {
            for (int a = 0; a < result[0].length; a++) {
                result[i][a] = raw[i][a] - another[i][a];
            }
        }

        return new Matrix(result);
    }

    public boolean isSquare() {
        return rows() == columns();
    }

    public int rows() {
        return raw.length;
    }

    public int columns() {
        return raw[0].length;
    }

    public double[][] raw() {
        return raw;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Matrix matrix) {
            if (matrix.rows() != rows() || matrix.columns() != columns()) {
                return false;
            }
            for (int i = 0; i < raw.length; i++) {
                for (int a = 0; a < raw[i].length; a++) {
                    if (matrix.raw()[i][a] != raw[i][a]) {
                        return false;
                    }
                }
            }
            return true;
        }
        return false;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("[").append("\n");
        for (int i = 0; i < raw.length; i++) {
            builder.append("    ").append("[");
            for (int a = 0; a < raw[i].length; a++) {
                builder.append(raw[i][a]);
                if (a != raw[i].length - 1) {
                    builder.append(", ");
                }
            }
            builder.append("]");
            if (i != raw.length - 1) {
                builder.append(",");
            }
            builder.append("\n");
        }
        return builder.append("]").toString();
    }

    @Override
    public Iterator<Double> iterator() {
        return new MatrixIterator(raw);
    }

    public static class MatrixIterator implements Iterator<Double> {

        private final double[] data;
        private int current;

        public MatrixIterator(double[][] data) {
            this.data = MathUtil.unwrap(data);
            this.current = 0;
        }

        @Override
        public boolean hasNext() {
            return current < data.length;
        }

        @Override
        public Double next() {
            if (!hasNext())
                throw new NoSuchElementException();
            return data[current++];
        }

    }

}
