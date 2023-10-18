package de.edux.data.handler;

import java.util.List;

public class DoNotHandleIncompleteRecords implements IIncompleteRecordsHandler {
  @Override
  public List<String[]> getCleanedDataset(List<String[]> dataset) {
    return dataset;
  }
}
