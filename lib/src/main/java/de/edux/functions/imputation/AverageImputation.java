package de.edux.functions.imputation;

import java.util.Arrays;
import java.util.List;

/**
 * Implements the {@code IImputationStrategy} interface to provide an average value imputation. This
 * strategy calculates the average of the non-missing numeric values in a column and substitutes the
 * missing values with this average.
 *
 * <p>It is important to note that this strategy is only applicable to columns with numeric data.
 * Attempting to use this strategy on categorical data will result in a {@code RuntimeException}.
 */
public class AverageImputation implements IImputationStrategy {
  /**
   * Performs average value imputation on the provided dataset column. Missing values are identified
   * as blank strings and are replaced by the average of the non-missing values. If the column
   * contains categorical data, a runtime exception is thrown.
   *
   * @param datasetColumn an array of {@code String} representing the column data with potential
   *     missing values.
   * @return an array of {@code String} where missing values have been imputed with the average of
   *     non-missing values.
   * @throws RuntimeException if the column data contains categorical values which cannot be
   *     averaged.
   */
  @Override
  public String[] performImputation(String[] datasetColumn) {
    checkIfColumnContainsCategoricalValues(datasetColumn);

    String[] updatedDatasetColumn = new String[datasetColumn.length];
    double average = calculateAverage(datasetColumn);

    for (int index = 0; index < datasetColumn.length; index++) {
      if (datasetColumn[index].isBlank()) {
        updatedDatasetColumn[index] = String.valueOf(average);

      } else {
        updatedDatasetColumn[index] = datasetColumn[index];
      }
    }

    return updatedDatasetColumn;
  }

  private void checkIfColumnContainsCategoricalValues(String[] datasetColumn) {
    for (String value : datasetColumn) {
      if (!isNumeric(value)) {
        throw new RuntimeException(
            "AVERAGE imputation strategy can not be used on categorical features. "
                + "Use MODE imputation strategy or perform a list wise deletion on the features.");
      }
    }
  }

  private boolean isNumeric(String value) {
    return value.matches("-?\\d+(\\.\\d+)?") || value.isBlank();
  }

  private double calculateAverage(String[] datasetColumn) {
    List<String> filteredDatasetColumn =
        Arrays.stream(datasetColumn).filter((value) -> !value.isBlank()).toList();
    int validValueCount = filteredDatasetColumn.size();
    double sumOfValidValues =
        filteredDatasetColumn.stream().map(Double::parseDouble).reduce(0.0, Double::sum);
    return sumOfValidValues / validValueCount;
  }
}
