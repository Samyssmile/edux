package de.edux.data.provider;

import java.io.File;

/**
 * The {@code Dataloader} interface defines a method for loading datasets from CSV files.
 * Implementations of this interface should handle the parsing of CSV files and configuration of
 * data processing according to the provided parameters.
 */
public interface Dataloader {
  /**
   * Loads a dataset from the specified CSV file, processes it, and returns a {@code DataProcessor}
   * that is ready to be used for further operations such as data manipulation or analysis.
   *
   * @param csvFile the CSV file to load the data from
   * @param csvSeparator the character that separates values in a row in the CSV file
   * @param skipHead a boolean indicating whether to skip the header row (true) or not (false)
   * @param inputColumns an array of indexes indicating which columns to include as input features
   * @param targetColumn the index of the column to use as the output label or target for
   *     predictions
   * @return a {@code DataProcessor} object that contains the processed data
   * @throws IllegalArgumentException if parameters are invalid or if the file cannot be processed
   */
  DataProcessor loadDataSetFromCSV(
      File csvFile, char csvSeparator, boolean skipHead, int[] inputColumns, int targetColumn);
}
