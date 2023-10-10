package de.edux.data.provider;

import de.edux.data.reader.CSVIDataReader;
import de.edux.data.reader.IDataReader;
import de.edux.ml.nn.network.api.Dataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class DataProcessor<T> extends DataPostProcessor<T> implements IDataUtil<T> {
    private static final Logger LOG = LoggerFactory.getLogger(DataProcessor.class);
    private final IDataReader csvDataReader;
    private ArrayList<T> dataset;
    private Dataset<T> splitedDataset;

    public DataProcessor() {
        this.csvDataReader = new CSVIDataReader();
    }

    public DataProcessor(IDataReader csvDataReader) {
        this.csvDataReader = csvDataReader;
    }

    @Override
    public List<T> loadDataSetFromCSV(File csvFile, char csvSeparator, boolean normalize, boolean shuffle, boolean filterIncompleteRecords) {
        List<String[]> x = csvDataReader.readFile(csvFile, csvSeparator);
        List<T> unmodifiableDataset = csvDataReader.readFile(csvFile, csvSeparator)
                .stream()
                .map(this::mapToDataRecord)
                .filter(record -> !filterIncompleteRecords || record != null)
                .toList();

        dataset = new ArrayList<>(unmodifiableDataset);
        LOG.info("Dataset loaded");

        if (normalize) {
            normalize(dataset);
            LOG.info("Dataset normalized");
        }

        if (shuffle) {
            Collections.shuffle(dataset);
            LOG.info("Dataset shuffled");
        }
        return dataset;
    }

    /**
     * Split data into train and test data
     *
     * @param data             data to split
     * @param trainTestSplitRatio ratio of train data
     * @return list of train and test data. First element is train data, second element is test data.
     */
    @Override
    public Dataset<T> split(List<T> data, double trainTestSplitRatio) {
        if (trainTestSplitRatio < 0.0 || trainTestSplitRatio > 1.0) {
            throw new IllegalArgumentException("Train-test split ratio must be between 0.0 and 1.0");
        }

        int trainSize = (int) (data.size() * trainTestSplitRatio);

        List<T> trainDataset = data.subList(0, trainSize);
        List<T> testDataset = data.subList(trainSize, data.size());

        splitedDataset = new Dataset<>(trainDataset, testDataset);
        return splitedDataset;
    }
    public ArrayList<T> getDataset() {
        return dataset;
    }

    public Dataset<T> getSplitedDataset() {
        return splitedDataset;
    }
}

