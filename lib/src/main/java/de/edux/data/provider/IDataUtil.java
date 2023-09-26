package de.edux.data.provider;

import java.io.File;
import java.util.List;

public interface IDataUtil<T> {
    List<T> loadTDataSet(File csvFile,  char csvSeparator, boolean normalize, boolean shuffle, boolean filterIncompleteRecords);

    List<List<T>> split(List<T> dataset, double trainTestSplitRatio);

    double[][] getInputs(List<T> dataset);

    double[][] getTargets(List<T> dataset);

}
