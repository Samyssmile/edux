package de.edux.data.provider;

import de.edux.functions.imputation.ImputationStrategy;

import java.util.List;
import java.util.Optional;

public interface DataPostProcessor {
    DataPostProcessor normalize();

    DataPostProcessor shuffle();

    DataPostProcessor imputation(String columnName, ImputationStrategy imputationStrategy);

    DataPostProcessor imputation(int columnIndex, ImputationStrategy imputationStrategy);

    List<String[]> getDataset();

    DataProcessor split(double splitRatio);


    public abstract Optional<Integer> getIndexOfColumn(String columnName);

    public abstract String[] getColumnDataOf(String columnName);

    public abstract String[] getColumnNames();

}
