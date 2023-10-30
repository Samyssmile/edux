package de.edux.data.provider;

import de.edux.functions.imputation.ImputationStrategy;

import java.util.List;
import java.util.Optional;

public interface DataPostProcessor {
  DataPostProcessor normalize();

  DataPostProcessor shuffle();

  void imputation(String columnName, ImputationStrategy imputationStrategy);

  void imputation(int columnIndex, ImputationStrategy imputationStrategy);

  void drop_incomplete_records();

  List<String[]> getDataset();

  DataProcessor split(double splitRatio);
}
