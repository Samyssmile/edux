package de.edux.functions.imputation;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class AverageImputationTest {
  @Test
  void performImputationWithCategoricalValuesShouldThrowRuntimeException() {
    String[] categoricalFeatures = {"A", "B", "C"};
    assertThrows(
        RuntimeException.class,
        () -> new AverageImputation().performImputation(categoricalFeatures));
  }

  @Test
  void performImputationWithNumericalValuesTest() {
    String[] numericalFeaturesWithMissingValues = {"1", "", "2", "3", "", "4"};
    AverageImputation imputter = new AverageImputation();
    String[] numericalFeaturesWithImputtedValues =
        imputter.performImputation(numericalFeaturesWithMissingValues);
    assertAll(
        () -> assertEquals("2.5", numericalFeaturesWithImputtedValues[1]),
        () -> assertEquals("2.5", numericalFeaturesWithImputtedValues[4]));
  }
}
