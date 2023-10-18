package de.edux.data.handler;

import java.util.ArrayList;
import java.util.List;

public class AverageFillIncompleteRecordsHandler implements IIncompleteRecordsHandler {
  @Override
  public List<String[]> getCleanedDataset(List<String[]> dataset) {
    List<String> typeOfFeatures = getFeatureTypes(dataset);
    List<String[]> cleanedDataset =
        dropRecordsWithIncompleteCategoricalFeature(dataset, typeOfFeatures);

    return averageFillRecordsWithIncompleteNumericalFeature(cleanedDataset, typeOfFeatures);
  }

  private List<String[]> averageFillRecordsWithIncompleteNumericalFeature(
      List<String[]> dataset, List<String> typeOfFeatures) {
    for (int columnIndex = 0; columnIndex < typeOfFeatures.size(); columnIndex++) {
      int validFeatureCount = 0;
      double sum = 0;
      double average;

      if (typeOfFeatures.get(columnIndex).equals("numerical")) {
        for (String[] record : dataset) {
          if (isCompleteFeature(record[columnIndex])) {
            validFeatureCount++;
            sum += Double.parseDouble(record[columnIndex]);
          }
        }

        if (validFeatureCount == 0) {
          continue;
        }

        average = sum / validFeatureCount;

        for (String[] record : dataset) {
          if (!isCompleteFeature(record[columnIndex])) {
            record[columnIndex] = String.valueOf(average);
          }
        }
      }
    }
    return dataset;
  }

  private List<String[]> dropRecordsWithIncompleteCategoricalFeature(
      List<String[]> dataset, List<String> typeOfFeatures) {

    for (int columnIndex = 0; columnIndex < typeOfFeatures.size(); columnIndex++) {
      if (typeOfFeatures.get(columnIndex).equals("categorical")) {
        int columnIndexFin = columnIndex;
        dataset =
            dataset.stream().filter(record -> isCompleteFeature(record[columnIndexFin])).toList();
      }
    }
    return dataset;
  }

  private List<String> getFeatureTypes(List<String[]> dataset) {
    List<String> featureTypes = new ArrayList<>();
    for (String[] record : dataset) {
      if (containsIncompleteFeature(record)) {
        continue;
      }
      for (String feature : record) {
        if (isNumeric(feature)) {
          featureTypes.add("numerical");
        } else {
          featureTypes.add("categorical");
        }
      }
      break;
    }
    return featureTypes;
  }

  private boolean isNumeric(String feature) {
    return feature.matches("-?\\d+(\\.\\d+)?");
  }

  private boolean isCompleteFeature(String feature) {
    return !feature.isBlank();
  }

  private boolean containsIncompleteFeature(String[] record) {
    for (String feature : record) {
      if (feature.isBlank()) {
        return true;
      }
    }
    return false;
  }
}
