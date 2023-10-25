package de.edux.data.provider;

import de.edux.functions.imputation.ImputationStrategy;

import java.util.List;

public interface DataPostProcessor {
    DataPostProcessor normalize();

    DataPostProcessor shuffle();

    DataPostProcessor imputation(String columnName, ImputationStrategy imputationStrategy);

    DataPostProcessor imputation(int columnIndex, ImputationStrategy imputationStrategy);

    List<String[]> getDataset();

    DataProcessor split(double splitRatio);


}
