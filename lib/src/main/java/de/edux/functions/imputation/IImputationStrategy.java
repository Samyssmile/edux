package de.edux.functions.imputation;

public interface IImputationStrategy {
  String[] performImputation(String[] columnData);
}
