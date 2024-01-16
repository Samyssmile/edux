package de.edux.functions.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

public class MinimaxTest {

  @Test
  public void testMinimax() {
    double[] emptyArray = {};
    double[] tooSmallArray = { 3.0 };
    double[] incorrectArraySize = { 1.0, 1.3, 2.5, 0.8, 1.6 };
    double[] size2Array = { 1.2, 0.9 };
    double[] sameValueArray = { 0.0, 0.000, 0.00, 0.0000 };
    double[] goodArray = { 1.8, 1.75, 2.3215, 1.681 };
    double[] largeArray = { 3.0, 5.0, 2.0, 9.0, 12.0, 5.0, 23.0, 22.0 };

    Minimax minimax = new Minimax();

    IllegalArgumentException noOutcomesException = assertThrows(IllegalArgumentException.class,
        () -> minimax.setOutcomes(emptyArray));
    IllegalArgumentException oneOutcomeException = assertThrows(IllegalArgumentException.class,
        () -> minimax.setOutcomes(tooSmallArray));
    IllegalArgumentException badOutcomeSizeException = assertThrows(IllegalArgumentException.class,
        () -> minimax.setOutcomes(incorrectArraySize));

    assertEquals("The number of outcomes must be greater than 1", noOutcomesException.getMessage());
    assertEquals("The number of outcomes must be greater than 1", oneOutcomeException.getMessage());
    assertEquals("The number of outcomes must be equal to a power of 2", badOutcomeSizeException.getMessage());

    assertEquals(1.2, minimax.setOutcomes(size2Array).execute());
    assertEquals(0.0, minimax.setOutcomes(sameValueArray).execute());
    assertEquals(1.75, minimax.setOutcomes(goodArray).execute());
    assertEquals(12.0, minimax.setOutcomes(largeArray).execute());
  }
}
