package de.edux.data.provider;

import de.edux.data.reader.IDataReader;
import de.edux.functions.imputation.ImputationStrategy;
import java.io.File;
import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataProcessor implements DataPostProcessor, Dataset, DataloaderV2 {
  private static final Logger LOG = LoggerFactory.getLogger(DataProcessor.class);
  private final IDataReader dataReader;
  private final Normalizer normalizer;
  private final Map<String, Integer> indexToClassMap = new HashMap<>();
  private String[] columnNames;
  private List<String[]> dataset;
  private List<String[]> trainData;
  private List<String[]> testData;

  public DataProcessor(IDataReader dataReader) {
    this.dataReader = dataReader;
    normalizer = new DataNormalizer();
  }

  @Override
  public DataProcessor split(double splitRatio) {
    int splitIndex = (int) (dataset.size() * splitRatio);
    trainData = dataset.subList(0, splitIndex);
    testData = dataset.subList(splitIndex, dataset.size());

    return this;
  }

  @Override
  public DataProcessor loadDataSetFromCSV(
      File csvFile, char csvSeparator, boolean skipHead, int[] inputColumns, int targetColumn) {
    dataset = dataReader.readFile(csvFile, csvSeparator);

    if (skipHead) {
      skipHead();
    }

    List<String> uniqueClasses = new ArrayList<>();
    for (String[] row : dataset) {
      String label = row[targetColumn];
      if (!uniqueClasses.contains(label)) {
        uniqueClasses.add(label);
      }
    }

    for (int i = 0; i < uniqueClasses.size(); i++) {
      indexToClassMap.put(uniqueClasses.get(i), i);
    }

    LOG.info("Dataset loaded");
    return this;
  }

  private void skipHead() {
    columnNames = dataset.remove(0);
  }

  @Override
  public DataPostProcessor normalize() {
    this.dataset = this.normalizer.normalize(dataset);
    return this;
  }

  @Override
  public DataPostProcessor shuffle() {
    Collections.shuffle(dataset);
    return this;
  }

  @Override
  public List<String[]> getDataset() {
    return dataset;
  }

  @Override
  public double[][] getInputs(List<String[]> dataset, int[] inputColumns) {
    if (dataset == null || dataset.isEmpty() || inputColumns == null || inputColumns.length == 0) {
      throw new IllegalArgumentException("Did you call split() before?");
    }

    int numRows = dataset.size();
    double[][] inputs = new double[numRows][inputColumns.length];

    for (int i = 0; i < numRows; i++) {
      String[] row = dataset.get(i);
      for (int j = 0; j < inputColumns.length; j++) {
        int colIndex = inputColumns[j];
        try {
          inputs[i][j] = Double.parseDouble(row[colIndex]);
        } catch (NumberFormatException e) {
          inputs[i][j] = 0;
        }
      }
    }

    return inputs;
  }

  @Override
  public double[][] getTargets(List<String[]> dataset, int targetColumn) {
    if (dataset == null || dataset.isEmpty()) {
      throw new IllegalArgumentException("Dataset darf nicht leer sein.");
    }

    double[][] targets = new double[dataset.size()][indexToClassMap.size()];
    for (int i = 0; i < dataset.size(); i++) {
      String value = dataset.get(i)[targetColumn];
      int index = indexToClassMap.get(value);
      targets[i][index] = 1.0;
    }

    return targets;
  }

  @Override
  public Map<String, Integer> getClassMap() {
    return indexToClassMap;
  }

  @Override
  public Optional<Integer> getIndexOfColumn(String columnName) {
    for (int i = 0; i < columnNames.length; i++) {
      if (columnNames[i].equals(columnName)) {
        return Optional.of(i);
      }
    }
    return Optional.empty();
  }

  public String[] getColumnDataOf(String columnName) {
    Optional<Integer> index = getIndexOfColumn(columnName);
    if (index.isEmpty()) {
      throw new IllegalArgumentException("Column name not found");
    }
    int columnIndex = index.get();
    String[] columnData = new String[dataset.size()];
    for (int i = 0; i < dataset.size(); i++) {
      columnData[i] = dataset.get(i)[columnIndex];
    }
    return columnData;
  }

  @Override
  public String[] getColumnNames() {
    return columnNames;
  }

  public void imputation(String columnName, ImputationStrategy imputationStrategy) {
    String[] columnDataToUpdate = getColumnDataOf(columnName);
    String[] updatedColumnData =
        imputationStrategy.getImputation().performImputation(columnDataToUpdate);
    int columnIndex = getIndexOfColumn(columnName).get();

    for (int row = 0; row < dataset.size(); row++) {
      dataset.get(row)[columnIndex] = updatedColumnData[row];
    }
  }

  public void imputation(int columnIndex, ImputationStrategy imputationStrategy) {
    String columnName = getColumnNames()[columnIndex];
    this.imputation(columnName, imputationStrategy);
  }

    @Override
    public void drop_incomplete_records() {
      dataset = dataset.stream().filter((record) -> {
          for (String feature: record) {
              if (feature.isBlank()) {
                  return false;
              }
          }
          return true;
      }).toList();
    }

    @Override
  public double[][] getTrainFeatures(int[] inputColumns) {
    return getInputs(trainData, inputColumns);
  }

  @Override
  public double[][] getTrainLabels(int targetColumn) {
    return getTargets(trainData, targetColumn);
  }

  @Override
  public double[][] getTestFeatures(int[] inputColumns) {
    return getInputs(testData, inputColumns);
  }

  @Override
  public double[][] getTestLabels(int targetColumn) {
    return getTargets(testData, targetColumn);
  }
}
