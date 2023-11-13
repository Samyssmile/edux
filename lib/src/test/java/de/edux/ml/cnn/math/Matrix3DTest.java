package de.edux.ml.cnn.math;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Matrix3DTest {

  private Matrix3D matrix;
  private Matrix3D otherMatrix;
  private Matrix3D kernel;

  @BeforeEach
  void setUp() {
    matrix = new Matrix3D(2, 3, 3); // Erstellen einer Matrix mit Dummy-Daten
    otherMatrix =
        new Matrix3D(2, 3, 3); // Eine andere Matrix für Operationen wie add, subtract, etc.
    kernel = new Matrix3D(2, 2, 2); // Kernel für convolve-Test
    // Hier könntest du deine Matrix mit spezifischen Werten initialisieren
  }

  @Test
  void testConvolve() {
    // Initialisieren der Matrix und des Kernels mit spezifischen Werten
    matrix.set(0, 0, 0, 1);
    matrix.set(0, 0, 1, 2);
    matrix.set(0, 1, 0, 3);
    matrix.set(0, 1, 1, 4);
    // Setze Werte für weitere Tiefen und Spalten, falls erforderlich

    kernel.set(0, 0, 0, 1);
    kernel.set(0, 0, 1, 0);
    kernel.set(0, 1, 0, 0);
    kernel.set(0, 1, 1, 1);
    // Setze Werte für weitere Tiefen und Spalten im Kernel, falls erforderlich

    // Führe die convolve Operation aus
    Matrix3D result = matrix.convolve(kernel, 1, 0);

    // Überprüfungen
    assertNotNull(result);
    assertEquals(3, result.getRows()); // Überprüfe die Anzahl der Zeilen im Ergebnis
    assertEquals(3, result.getCols()); // Überprüfe die Anzahl der Spalten im Ergebnis

    // Überprüfe spezifische Werte in der Ergebnismatrix
    assertEquals(1, result.get(0, 0, 0));
    assertEquals(2, result.get(0, 0, 1));
    // Füge weitere Überprüfungen hinzu, um sicherzustellen, dass die Faltung korrekt durchgeführt
    // wurde
  }

  @Test
  void testMaxPooling() {
    // Initialisieren der Matrix mit spezifischen Werten
    matrix.set(0, 0, 0, 1);
    matrix.set(0, 0, 1, 2);
    matrix.set(0, 0, 2, 3);
    matrix.set(0, 1, 0, 4);
    matrix.set(0, 1, 1, 5);
    matrix.set(0, 1, 2, 6);
    matrix.set(0, 2, 0, 7);
    matrix.set(0, 2, 1, 8);
    matrix.set(0, 2, 2, 9);
    // Füge Werte für weitere Tiefen hinzu, falls erforderlich

    // Führe Max Pooling aus
    Matrix3D result = matrix.maxPooling(2, 2);

    // Überprüfungen
    assertNotNull(result);
    assertEquals(1, result.getDepth()); // Überprüfe die Tiefe des Ergebnisses
    assertEquals(2, result.getRows()); // Überprüfe die Anzahl der Zeilen im Ergebnis
    assertEquals(2, result.getCols()); // Überprüfe die Anzahl der Spalten im Ergebnis

    // Überprüfe spezifische Werte in der Ergebnismatrix
    assertEquals(5, result.get(0, 0, 0)); // Der maximale Wert im ersten 2x2 Block
    assertEquals(6, result.get(0, 0, 1)); // Der maximale Wert im zweiten 2x2 Block
    assertEquals(8, result.get(0, 1, 0)); // Der maximale Wert im dritten 2x2 Block
    assertEquals(9, result.get(0, 1, 1)); // Der maximale Wert im vierten 2x2 Block
  }

  @Test
  void testApplyReLU() {
    Matrix3D result = matrix.applyReLU();
    assertNotNull(result);
    // Überprüfe, ob alle negativen Werte in result 0 sind
  }

  @Test
  void testApplyLeakyReLU() {
    Matrix3D result = matrix.applyLeakyReLU();
    assertNotNull(result);
    // Überprüfe die Anwendung von LeakyReLU
  }

  @Test
  void testConvolveBackprop() {
    Matrix3D gradient = new Matrix3D(2, 3, 3); // Gradientenmatrix
    Matrix3D result = matrix.convolveBackprop(gradient, 1, 0);
    assertNotNull(result);
    // Überprüfe spezifische Eigenschaften von result
  }

  @Test
  void testAdd() {
    Matrix3D result = matrix.add(otherMatrix);
    assertNotNull(result);
    // Überprüfe die Addition
  }

  @Test
  void testSubtract() {
    Matrix3D result = matrix.subtract(otherMatrix);
    assertNotNull(result);
    // Überprüfe die Subtraktion
  }

  @Test
  void testMultiplyElementWise() {
    Matrix3D result = matrix.multiplyElementWise(otherMatrix);
    assertNotNull(result);
    // Überprüfe die elementweise Multiplikation
  }

  @Test
  void testNormalize() {
    matrix.normalize(0.5, 2.0); // Mittelwert und Standardabweichung als Beispiel
    // Überprüfe, ob matrix normalisiert wurde
  }
}
