package de.edux.data.provider;

import java.util.List;

public abstract class DataPostProcessor<T> {
    public abstract void normalize(List<T> rowDataset);
    public abstract T mapToDataRecord(String[] csvLine);
    public abstract double[][] getInputs(List<T> dataset);
    public abstract double[][] getTargets(List<T> dataset);

}
