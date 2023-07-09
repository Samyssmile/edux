package de.edux.data.provider;

import java.util.List;

public interface IDataProvider<T> {
    List<T> getTrainData();

    List<T> getTestData();

    void printStatistics();

    T getRandom(boolean equalDistribution);

    double[][] getFeatures();

    double[][] getLabels();

    double[][] getFeaturesTest();

    double[][] getLabelsTest();

    String getDescription();

}
