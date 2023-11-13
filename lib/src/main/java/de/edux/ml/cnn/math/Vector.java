package de.edux.ml.cnn.math;

import java.util.Random;

public class Vector {
    private double[] data;

    public Vector(int size) {
        data = new double[size];
        // Random initialisieren (optional)
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            data[i] = random.nextDouble(); // oder 0 für Initialisierung mit Null
        }
    }

    // Vektor-Skalar Multiplikation
    public Vector multiply(double scalar) {
        Vector result = new Vector(this.data.length);
        for (int i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        return result;
    }

    // Vektor-Vektor Multiplikation (elementweise)
    public Vector multiply(Vector other) {
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException("Vektoren ungleicher Länge");
        }

        Vector result = new Vector(this.data.length);
        for (int i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    // Äußeres Produkt
    public Matrix3D outerProduct(Vector other) {
        Matrix3D matrix = new Matrix3D(this.data.length, other.data.length, 1);
        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < other.data.length; j++) {
                matrix.set(i, j, 0, this.data[i] * other.data[j]);
            }
        }
        return matrix;
    }

    // Subtraktion eines anderen Vektors
    public Vector subtract(Vector other) {
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException("Vektoren ungleicher Länge");
        }

        Vector result = new Vector(this.data.length);
        for (int i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    // Hinzufügen eines anderen Vektors
    public Vector add(Vector other) {
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException("Vektoren ungleicher Länge");
        }

        Vector result = new Vector(this.data.length);
        for (int i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    // Konvertierung in eine Matrix3D
    public Matrix3D toMatrix3D() {
        Matrix3D matrix = new Matrix3D(1, this.data.length, 1);
        for (int i = 0; i < this.data.length; i++) {
            matrix.set(0, i, 0, this.data[i]);
        }
        return matrix;
    }

    // Getter und Setter
    public double[] getData() {
        return data;
    }

    public void setData(double[] data) {
        this.data = data;
    }
}
