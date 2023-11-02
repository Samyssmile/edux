package de.edux.functions.imputation;

import java.util.*;

/**
 * Implements the {@code IImputationStrategy} interface to provide a mode value imputation. This
 * strategy finds the most frequently occurring value, or mode, in a dataset column and substitutes
 * missing values with this mode.
 */
public class ModeImputation implements IImputationStrategy {

  /**
   * Performs mode value imputation on the provided dataset column. Missing values are identified as
   * blank strings and are replaced by the mode of the non-missing values. If multiple modes are
   * found, the first encountered in the dataset is used.
   *
   * @param datasetColumn an array of {@code String} representing the column data with potential
   *     missing values.
   * @return an array of {@code String} where missing values have been imputed with the mode of
   *     non-missing values.
   */
  @Override
  public String[] performImputation(String[] datasetColumn) {

    String[] updatedDatasetColumn = new String[datasetColumn.length];
    String modalValue = getModalValue(datasetColumn);

    for (int index = 0; index < datasetColumn.length; index++) {
      if (datasetColumn[index].isBlank()) {
        updatedDatasetColumn[index] = modalValue;
      } else {
        updatedDatasetColumn[index] = datasetColumn[index];
      }
    }

    return updatedDatasetColumn;
  }

  private String getModalValue(String[] datasetColumn) {
    String modalValue = null;
    int modalValueCount = 0;
    var uniqueValueFrequencyMap = getUniqueValueFrequencyMap(datasetColumn);
    Set<String> uniqueValues = uniqueValueFrequencyMap.keySet();

    for (String uniqueValue : uniqueValues) {
      if (modalValue == null) {
        modalValue = uniqueValue;
      } else if (uniqueValueFrequencyMap.get(uniqueValue) > modalValueCount) {
        modalValue = uniqueValue;
        modalValueCount = uniqueValueFrequencyMap.get(uniqueValue);
      }
    }

    return modalValue;
  }

  private Map<String, Integer> getUniqueValueFrequencyMap(String[] datasetColumn) {
    Map<String, Integer> uniqueValueFrequencyMap = new HashMap<>();
    for (String value : datasetColumn) {
      uniqueValueFrequencyMap.merge(value, 1, Integer::sum);
    }
    return uniqueValueFrequencyMap;
  }
}
