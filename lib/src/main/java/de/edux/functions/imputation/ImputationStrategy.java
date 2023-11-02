package de.edux.functions.imputation;

/**
 * Enumerates the available imputation strategies for handling missing values in datasets. Each
 * strategy is associated with a concrete implementation of {@code IImputationStrategy} that defines
 * the specific imputation behavior.
 */
public enum ImputationStrategy {
  /**
   * Imputation strategy that replaces missing values with the average of the non-missing values in
   * the dataset column. This strategy is suitable for numerical data only.
   */
  AVERAGE(new AverageImputation()),

  /**
   * Imputation strategy that replaces missing values with the most frequently occurring value
   * (mode) in the dataset column. This strategy can be used for both numerical and categorical
   * data.
   */
  MODE(new ModeImputation());

  private final IImputationStrategy imputation;

  /**
   * Constructor for the enum that associates each imputation strategy with a concrete {@code
   * IImputationStrategy} implementation.
   *
   * @param imputation the imputation strategy implementation
   */
  ImputationStrategy(IImputationStrategy imputation) {
    this.imputation = imputation;
  }

  /**
   * Retrieves the {@code IImputationStrategy} implementation associated with the imputation
   * strategy.
   *
   * @return the imputation strategy implementation
   */
  public IImputationStrategy getImputation() {
    return this.imputation;
  }
}
