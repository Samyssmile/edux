package de.edux.functions.imputation;

import java.util.*;

public class ModeImputation implements IImputationStrategy {
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
