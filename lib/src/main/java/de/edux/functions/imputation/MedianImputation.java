package de.edux.functions.imputation;

import java.util.Arrays;

/**
 * Implements the {@code IImputationStrategy} interface to provide a median value imputation. This
 * strategy calculates the median of the non-missing numeric values in a column and substitutes the
 * missing values with this median.
 *
 * <p>It is important to note that this strategy is only applicable to columns with numeric data.
 * Attempting to use this strategy on categorical data will result in a {@code RuntimeException}.
 */
public class MedianImputation implements IImputationStrategy {
  @Override
  public String[] performImputation(String[] datasetColumn) {
    checkIfColumnContainsCategoricalValues(datasetColumn);

    String[] updatedDatasetColumn = new String[datasetColumn.length];
    double median = calculateMedian(datasetColumn);

    for (int index = 0; index < datasetColumn.length; index++) {
      if (datasetColumn[index].isBlank()) {
        updatedDatasetColumn[index] = String.valueOf(median);

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
            "MEDIAN imputation strategy can not be used on categorical features. "
                + "Use MODE imputation strategy or perform a list wise deletion on the features.");
      }
    }
  }

  private boolean isNumeric(String value) {
    return value.matches("-?\\d+(\\.\\d+)?") || value.isBlank();
  }

  double calculateMedian(String[] datasetColumn) {
   double[] filteredDatasetColumnInNumbers = Arrays.stream(datasetColumn)
           .filter(value -> !value.isBlank())
           .mapToDouble(Double::parseDouble)
           .sorted()
           .toArray();
    if (filteredDatasetColumnInNumbers.length % 2 == 0) {
      Double upper = filteredDatasetColumnInNumbers[filteredDatasetColumnInNumbers.length / 2];
      Double lower =
          filteredDatasetColumnInNumbers[(filteredDatasetColumnInNumbers.length / 2) - 1];
      return (upper + lower) / 2.0;
    }
    return filteredDatasetColumnInNumbers[filteredDatasetColumnInNumbers.length / 2];
  }
}
