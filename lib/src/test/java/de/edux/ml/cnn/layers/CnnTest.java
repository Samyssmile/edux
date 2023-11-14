package de.edux.ml.cnn.layers;

import static org.junit.jupiter.api.Assertions.assertEquals;

import de.edux.ml.cnn.math.Matrix3D;
import org.junit.jupiter.api.Test;

class CnnTest {

  @Test
  void denseLayer() {

    DenseLayer denseLayer = new DenseLayer(1568, 10, 1);

    Matrix3D input = new Matrix3D(1, 1, 1568);

    Matrix3D output = denseLayer.forward(input);

    Matrix3D backwarinInput = denseLayer.backward(output, 0.1);

    assertEquals(input, backwarinInput);
  }

  @Test
  void convolutionalLayer() {
    ConvolutionalLayer conv = new ConvolutionalLayer(8, 3, 1, 1, 1);
    Matrix3D input = new Matrix3D(1, 28, 28);

    Matrix3D output = conv.forward(input);
    Matrix3D backwarinInput = conv.backward(output, 0.1);

    assertEquals(input, backwarinInput);
  }

  @Test
  void maxPoolingLayer() {
    MaxPoolingLayer poolingLayer = new MaxPoolingLayer(2, 2);
    Matrix3D input = new Matrix3D(8, 28, 28);

    Matrix3D output = poolingLayer.forward(input);
    Matrix3D backwarinInput = poolingLayer.backward(output, 0.1);

    assertEquals(input, backwarinInput);
  }

  @Test
  void reluLayer() {
    ReLuLayer reLuLayer = new ReLuLayer();
    Matrix3D input = new Matrix3D(8, 28, 28);

    Matrix3D output = reLuLayer.forward(input);
    Matrix3D backwarinInput = reLuLayer.backward(output, 0.01);

    assertEquals(input, backwarinInput);
  }

  @Test
  void flattenLayer() {
    FlattenLayer flattenLayer = new FlattenLayer();
    Matrix3D input = new Matrix3D(8, 28, 28);

    Matrix3D output = flattenLayer.forward(input);
    Matrix3D backwarinInput = flattenLayer.backward(output, 0.01);

    assertEquals(input, backwarinInput);
  }

  @Test
  void softmaxLayer() {
    SoftmaxLayer reLuLayer = new SoftmaxLayer();
    Matrix3D input = new Matrix3D(1, 1, 10);

    Matrix3D output = reLuLayer.forward(input);
    Matrix3D backwarinInput = reLuLayer.backward(output, 0.01);

    assertEquals(input, backwarinInput);
  }
}
