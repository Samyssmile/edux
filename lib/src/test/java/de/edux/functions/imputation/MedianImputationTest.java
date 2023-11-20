package de.edux.functions.imputation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MedianImputationTest {
    @Test
    void performImputationWithCategoricalValuesShouldThrowRuntimeException() {
        String[] categoricalFeatures = {"A", "B", "C"};
        assertThrows(
                RuntimeException.class,
                () -> new MedianImputation().performImputation(categoricalFeatures));
    }

    @Test
    void performImputationWithNumericalValuesTest() {
        String[] numericalFeaturesWithMissingValues = {"1", "", "2", "3", "", "4"};
        MedianImputation imputter = new MedianImputation();
        String[] numericalFeaturesWithImputtedValues =
                imputter.performImputation(numericalFeaturesWithMissingValues);
        assertAll(
                () -> assertEquals("2.5", numericalFeaturesWithImputtedValues[1]),
                () -> assertEquals("2.5", numericalFeaturesWithImputtedValues[4]));
    }
}
