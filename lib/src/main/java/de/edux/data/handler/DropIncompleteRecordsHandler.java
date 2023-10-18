package de.edux.data.handler;

import java.util.List;

public class DropIncompleteRecordsHandler implements IIncompleteRecordsHandler {
  @Override
  public List<String[]> getCleanedDataset(List<String[]> dataset) {

    return dataset.stream().filter(this::containsOnlyCompletedFeatures).toList();
  }

  private boolean containsOnlyCompletedFeatures(String[] record) {
    for (String feature : record) {
      if (feature.isBlank()) {
        return false;
      }
    }
    return true;
  }
}
