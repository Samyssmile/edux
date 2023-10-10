package de.edux.data.provider;

import de.edux.ml.nn.network.api.Dataset;

import java.io.File;
import java.util.List;

public interface IDataUtil<T> {
    List<T> loadDataSetFromCSV(File csvFile,  char csvSeparator, boolean normalize, boolean shuffle, boolean filterIncompleteRecords);

    Dataset<T> split(List<T> dataset, double trainTestSplitRatio);

    double[][] getInputs(List<T> dataset);

    double[][] getTargets(List<T> dataset);

}
