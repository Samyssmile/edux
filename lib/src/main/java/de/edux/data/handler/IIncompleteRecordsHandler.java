package de.edux.data.handler;

import java.util.List;

public interface IIncompleteRecordsHandler {
  List<String[]> getCleanedDataset(List<String[]> dataset);
}
