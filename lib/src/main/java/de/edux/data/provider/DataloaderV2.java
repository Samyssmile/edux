package de.edux.data.provider;

import java.io.File;

public interface DataloaderV2 {
  DataProcessor loadDataSetFromCSV(
      File csvFile, char csvSeparator, boolean skipHead, int[] inputColumns, int targetColumn);
}
