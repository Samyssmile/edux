package de.edux.data.provider;

import de.edux.data.reader.CSVIDataReader;
import de.edux.data.reader.IDataReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class DataUtil<T> extends DataPostProcessor<T> implements IDataUtil<T> {
    private static final Logger logger = LoggerFactory.getLogger(DataUtil.class);
    private final IDataReader csvDataReader;
    public DataUtil() {
        this.csvDataReader = new CSVIDataReader();
    }

    public DataUtil(IDataReader csvDataReader) {
        this.csvDataReader = csvDataReader;
    }

    @Override
    public List<T> loadTDataSet(File csvFile, char csvSeparator, boolean normalize, boolean shuffle, boolean filterIncompleteRecords) {
        List<String[]> x = csvDataReader.readFile(csvFile, csvSeparator);
        List<T> unmodifiableDataset = csvDataReader.readFile(csvFile, csvSeparator)
                .stream()
                .map(this::mapToDataRecord)
                .filter(record -> !filterIncompleteRecords || record != null)
                .toList();

        List<T> dataset = new ArrayList<>(unmodifiableDataset);
        logger.info("Dataset loaded");

        if (normalize) {
            normalize(dataset);
            logger.info("Dataset normalized");
        }

        if (shuffle) {
            Collections.shuffle(dataset);
            logger.info("Dataset shuffled");
        }
        return dataset;
    }

    /**
     * Split dataset into train and test dataset
     *
     * @param dataset             dataset to split
     * @param trainTestSplitRatio ratio of train dataset
     * @return list of train and test dataset. First element is train dataset, second element is test dataset.
     */
    @Override
    public List<List<T>> split(List<T> dataset, double trainTestSplitRatio) {
        if (trainTestSplitRatio < 0.0 || trainTestSplitRatio > 1.0) {
            throw new IllegalArgumentException("Train-test split ratio must be between 0.0 and 1.0");
        }

        int trainSize = (int) (dataset.size() * trainTestSplitRatio);

        List<T> trainDataset = dataset.subList(0, trainSize);
        List<T> testDataset = dataset.subList(trainSize, dataset.size());

        return List.of(trainDataset, testDataset);
    }

}

