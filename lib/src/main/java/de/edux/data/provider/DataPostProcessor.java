package de.edux.data.provider;

import de.edux.functions.imputation.ImputationStrategy;
import java.util.List;

/**
 * The {@code DataPostProcessor} interface defines a set of methods for post-processing data. This
 * typically includes normalizing data, shuffling, handling missing values, and splitting datasets.
 * Implementations should ensure that the data is properly processed to be ready for subsequent
 * analysis or machine learning tasks.
 */
public interface DataPostProcessor {

  /**
   * Normalizes the dataset. This typically involves scaling the values of numeric attributes so
   * that they share a common scale, often between 0 and 1, without distorting differences in the
   * ranges of values.
   *
   * @return the {@code DataPostProcessor} instance with normalized data for method chaining
   */
  DataPostProcessor normalize();

  /**
   * Shuffles the dataset randomly. This is usually done to ensure that the data does not carry any
   * inherent bias in the order it was collected or presented.
   *
   * @return the {@code DataPostProcessor} instance with shuffled data for method chaining
   */
  DataPostProcessor shuffle();

  /**
   * Performs imputation on missing values in a specified column index using the provided imputation
   * strategy.
   *
   * @param columnIndex the index of the column to apply imputation
   * @param imputationStrategy the strategy to use for imputing missing values
   * @return the {@code DataPostProcessor} instance with imputed data for method chaining
   */
  DataPostProcessor imputation(int columnIndex, ImputationStrategy imputationStrategy);

  /**
   * Performs list-wise deletion on the dataset. This involves removing any rows with missing values
   * to ensure the dataset is complete. This method modifies the dataset in place and does not
   * return a value.
   */
  void performListWiseDeletion();

  /**
   * Retrieves the processed dataset as a list of string arrays. Each string array represents a row
   * in the dataset.
   *
   * @return a list of string arrays representing the dataset
   */
  List<String[]> getDataset();

  /**
   * Splits the dataset into two separate datasets according to the specified split ratio. The split
   * ratio determines the proportion of data to be used for the first dataset (e.g., training set).
   *
   * @param splitRatio the ratio for splitting the dataset, where 0 < splitRatio < 1
   * @return a {@code DataProcessor} instance containing the first portion of the dataset according
   *     to the split ratio
   * @throws IllegalArgumentException if the split ratio is not between 0 and 1
   */
  DataProcessor split(double splitRatio);
}
