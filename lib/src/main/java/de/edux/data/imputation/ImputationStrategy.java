package de.edux.data.imputation;

public enum ImputationStrategy {
  AVERAGE(new AverageImputation()),
  MODE(new ModeImputation());

  private final IImputationStrategy imputation;

  ImputationStrategy(IImputationStrategy imputation) {
    this.imputation = imputation;
  }

  public IImputationStrategy getImputation() {
    return this.imputation;
  }
}
