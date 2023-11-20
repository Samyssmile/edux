package de.edux.functions.imputation;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

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

  @Test
  public void testCalculateMedianWithLargeDataset() {
    String[] largeDataset = new String[1000000];
    Random random = new Random();
    for (int i = 0; i < largeDataset.length; i++) {
      if (random.nextDouble() < 0.05) { // 5% empty values
        largeDataset[i] = "";
      } else {
        largeDataset[i] = String.valueOf(random.nextDouble() * 1000000);
      }
    }

    // Erwarteter Median
    double[] numericValues =
        Arrays.stream(largeDataset)
            .filter(s -> !s.isBlank())
            .mapToDouble(Double::parseDouble)
            .sorted()
            .toArray();
    double expectedMedian =
        numericValues.length % 2 == 0
            ? (numericValues[numericValues.length / 2]
                    + numericValues[numericValues.length / 2 - 1])
                / 2.0
            : numericValues[numericValues.length / 2];

    MedianImputation medianImputation = new MedianImputation();

    long startTime = System.nanoTime();
    double calculatedMedian = medianImputation.calculateMedian(largeDataset);
    long endTime = System.nanoTime();

    System.out.println("Process time in seconds: " + (endTime - startTime) / 1e9);

    assertEquals(
        expectedMedian,
        calculatedMedian,
        0.001,
        "Calculated median should be equal to the expected median.");
  }
}
