package de.edux.data.imputation;

public interface IImputationStrategy {
  String[] performImputation(String[] columnData);
}
