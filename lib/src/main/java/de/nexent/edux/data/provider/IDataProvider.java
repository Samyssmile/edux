package de.nexent.edux.data.provider;

import java.util.List;

public interface IDataProvider<T> {
    List<T> getTrainData();

    List<T> getTestData();

    void printStatistics();

    T getRandom(boolean equalDistribution);

    double[][] getTrainFeatures();

    double[][] getTrainLabels();

    double[][] getTestFeatures();

    double[][] getTestLabels();

    String getDescription();

}
