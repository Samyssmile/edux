package de.edux.functions.imputation;

/**
 * Defines a strategy interface for imputing missing values within a column of data. Implementations
 * of this interface should provide a concrete imputation method that can handle various types of
 * missing data according to specific rules or algorithms.
 */
public interface IImputationStrategy {

  /**
   * Performs imputation on the provided column data array. Missing values within the array are
   * expected to be filled with substituted values determined by the specific imputation strategy
   * implemented.
   *
   * @param columnData an array of {@code String} representing the data of a single column, where
   *     missing values are to be imputed.
   * @return an array of {@code String} representing the column data after imputation has been
   *     performed.
   */
  String[] performImputation(String[] columnData);
}
