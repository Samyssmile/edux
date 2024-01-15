package de.edux.functions.ai;

import java.util.Arrays;

/**
 * Implements the Minimax algorithm. The user must first set the outcomes with
 * the setOutcomes(...) method and then execute the algorithm with the execute()
 * method.
 */
public class Minimax {
  private double[] outcomes;

  /**
   * Takes the possible outcomes for the Minimax algorithm to later execute on.
   * The outcomes are validated against 2 conditions: (1) the amount of
   * possibilities is at least 2 and (2) the amount of possibilities is a power of
   * 2. An illegal argument exception is thrown if both conditions are not met.
   * This method returns the minimax object
   *
   * @param outcomes
   * @return minimax
   */
  public Minimax setOutcomes(double... outcomes) {
    if (outcomes.length < 2) {
      throw new IllegalArgumentException("The number of outcomes must be greater than 1");
    }
    else if ((outcomes.length & (outcomes.length - 1)) != 0) {
      System.out.println(outcomes.length & (outcomes.length + 1));
      throw new IllegalArgumentException("The number of outcomes must be equal to a power of 2");
    }
    this.outcomes = outcomes;
    return this;
  }

  /**
   * Executes the Minimax algorithm on already validated outcomes. Returns the
   * double that will result from the algorithm.
   *
   * @return double
   */
  public double execute() {
    return this.maximum(this.outcomes);
  }

  /**
   * if only 2 options are available, returns the maximum value otherwise calls
   * the minimum on the first and second half of the outcomes.
   *
   * @param outcomes
   * @return double
   */
  private double maximum(double... outcomes) {
    System.out.println("Outcome length in maximum: " + outcomes.length);
    if (outcomes.length == 2) {
      return Math.max(outcomes[0], outcomes[1]);
    } else {
      return Math.max(this.minimum(Arrays.copyOfRange(outcomes, 0, outcomes.length / 2)),
          this.minimum(Arrays.copyOfRange(outcomes, outcomes.length / 2, outcomes.length)));
    }
  }

  /**
   * if only 2 options are available, returns the minimum value otherwise calls
   * the maximum on the first and second half of the outcomes.
   *
   * @param outcomes
   * @return double
   */
  private double minimum(double... outcomes) {
    System.out.println("Outcome length in minimum: " + outcomes.length);
    if (outcomes.length == 2) {
      return Math.min(outcomes[0], outcomes[1]);
    } else {
      return Math.min(this.maximum(Arrays.copyOfRange(outcomes, 0, outcomes.length / 2)),
          this.maximum(Arrays.copyOfRange(outcomes, outcomes.length / 2, outcomes.length)));
    }
  }
}
