package de.edux.data.handler;

import java.util.List;

public class DropIncompleteRecordsHandler implements IIncompleteRecordsHandler {
  @Override
  public List<String[]> getCleanedDataset(List<String[]> dataset) {
    List<String[]> cleanedDataset =
        dataset.stream().filter(this::containsOnlyCompletedFeatures).toList();

    if (cleanedDataset.size() < dataset.size() * 0.5) {
      throw new RuntimeException(
          "More than 50% of the records will be dropped with this IncompleteRecordsHandlerStrategy. "
              + "Consider using another IncompleteRecordsHandlerStrategy or handle this exception.");
    }

    return cleanedDataset;
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
