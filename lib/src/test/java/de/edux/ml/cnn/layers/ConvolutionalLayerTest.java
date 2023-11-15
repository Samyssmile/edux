package de.edux.ml.cnn.layers;

import static org.junit.jupiter.api.Assertions.*;

import de.edux.ml.cnn.math.Matrix3D;
import org.junit.jupiter.api.Test;

class ConvolutionalLayerTest {

  @Test
  void testForward() {
    // Initialisierung des ConvolutionalLayer
    ConvolutionalLayer layer =
        new ConvolutionalLayer(
            1, 3, 1, 0, 1); // 1 Filter, Filtergröße 3x3, Stride 1, kein Padding, Tiefe 1

    // Initialisierung der Filtergewichte für den Test (normalerweise zufällig)
    // Hier setzen wir die Filter manuell für den Test
    double[][][] filter = {
      {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}}
    }; // Einfacher vertikaler Kantendetektor
    layer.setFilters(
        new double[][][][] {filter}); // Methode setFilters muss in Ihrer Klasse implementiert sein

    // Vorbereitung der Testdaten
    Matrix3D inputData = new Matrix3D(1, 3, 3); // Tiefe 1, 3x3-Matrix
    inputData.addToValue(0, 0, 0, 1);
    inputData.set(0, 0, 0, 1);
    inputData.set(0, 0, 1, 2);
    inputData.set(0, 0, 2, 1);
    inputData.set(0, 1, 0, 0);
    inputData.set(0, 1, 1, 2);
    inputData.set(0, 1, 2, 0);
    inputData.set(0, 2, 0, 1);
    inputData.set(0, 2, 1, 0);
    inputData.set(0, 2, 2, 1);

    // Ausführen der forward-Methode
    Matrix3D output = layer.forward(inputData);

    // Überprüfen der Ergebnisse
    // Wir erwarten spezifische Werte basierend auf unserem Filter und den Eingabedaten
    assertEquals(2, output.get(0, 0, 0));
    assertEquals(2, output.get(0, 0, 1));
    assertEquals(-2, output.get(0, 0, 2));
    // Fügen Sie weitere Überprüfungen hinzu, um sicherzustellen, dass alle Ausgabewerte korrekt
  }

  @Test
  void testSimpleForward() {
    ConvolutionalLayer layer =
        new ConvolutionalLayer(
            1, 1, 1, 0, 1); // Einfacher 1x1-Filter, Stride 1, kein Padding, Tiefe 1
    double[][][] filter = {{{1}}}; // Sehr einfacher Filter
    layer.setFilters(new double[][][][] {filter});

    Matrix3D inputData = new Matrix3D(1, 3, 3);
    // Setzen Sie Werte in inputData...

    Matrix3D output = layer.forward(inputData);

    // Überprüfen Sie die Ergebnisse mit sehr einfachen Erwartungen
    assertEquals(inputData.get(0, 0, 0), output.get(0, 0, 0));
    // Weitere Überprüfungen...
  }
}
