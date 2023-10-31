package de.edux.functions.imputation;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ModeImputationTest {

  @Test
  void performImputationWithNumericalValuesTest() {
    String[] numericalFeaturesWithMissingValues = {"1", "", "1", "2", "", "3"};
    ModeImputation imputter = new ModeImputation();
    String[] numericalFeaturesWithImputtedValues =
        imputter.performImputation(numericalFeaturesWithMissingValues);
    assertAll(
        () -> assertEquals("1", numericalFeaturesWithImputtedValues[1]),
        () -> assertEquals("1", numericalFeaturesWithImputtedValues[4]));
  }

  @Test
  void performImputationWithCategoricalValuesTest() {
    String[] numericalFeaturesWithMissingValues = {"A", "", "A", "B", "", "C"};
    ModeImputation imputter = new ModeImputation();
    String[] numericalFeaturesWithImputtedValues =
        imputter.performImputation(numericalFeaturesWithMissingValues);
    assertAll(
        () -> assertEquals("A", numericalFeaturesWithImputtedValues[1]),
        () -> assertEquals("A", numericalFeaturesWithImputtedValues[4]));
  }
}
