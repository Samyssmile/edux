package de.example.data.iris;

import de.edux.data.provider.IDataProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static de.example.data.iris.IrisDataUtil.loadIrisDataSet;

public class IrisProvider implements IDataProvider<Iris> {
    private static final Logger LOG = LoggerFactory.getLogger(IrisProvider.class);
    private final List<Iris> dataset;
    private final List<List<Iris>> split;
    private final List<Iris> trainingData;
    private final List<Iris> testData;

    /**
     * @param normalize           Will normalize the data if true
     * @param shuffle             Will shuffle the data if true
     * @param trainTestSplitRatio Ratio of train and test data e.g. 0.8 means 80% train data and 20% test data
     */
    public IrisProvider(boolean normalize, boolean shuffle, double trainTestSplitRatio) {
        dataset = loadIrisDataSet(normalize, shuffle);
        split = IrisDataUtil.split(dataset, trainTestSplitRatio);
        trainingData = split.get(0);
        testData = split.get(1);
    }

    @Override
    public List<Iris> getTrainData() {
        return trainingData;
    }

    @Override
    public List<Iris> getTestData() {
        return testData;
    }

    @Override
    public void printStatistics() {
        LOG.info("========================= Data Statistic ==================");
        LOG.info("Total dataset size: " + dataset.size());
        LOG.info("Training dataset size: " + trainingData.size());
        LOG.info("Test data set size: " + testData.size());
        LOG.info("Classes: " + getTrainLabels()[0].length);
        LOG.info("===========================================================");
    }

    @Override
    public Iris getRandom(boolean equalDistribution) {
        return dataset.get((int) (Math.random() * dataset.size()));
    }

    @Override
    public double[][] getTrainFeatures() {
        return featuresOf(trainingData);
    }

    @Override
    public double[][] getTrainLabels() {
        return labelsOf(trainingData);
    }

    @Override
    public double[][] getTestLabels() {
        return labelsOf(testData);
    }

    @Override
    public double[][] getTestFeatures() {
        return featuresOf(testData);
    }

    private double[][] featuresOf(List<Iris> testData) {
        double[][] features = new double[testData.size()][4];
        for (int i = 0; i < testData.size(); i++) {
            features[i][0] = testData.get(i).sepalLength;
            features[i][1] = testData.get(i).sepalWidth;
            features[i][2] = testData.get(i).petalLength;
            features[i][3] = testData.get(i).petalWidth;
        }
        return features;
    }
    private double[][] labelsOf(List<Iris> data) {
        double[][] labels = new double[data.size()][3];
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).variety.equals("Setosa")) {
                labels[i][0] = 1;
                labels[i][1] = 0;
                labels[i][2] = 0;
            }
            if (data.get(i).variety.equals("Versicolor")) {
                labels[i][0] = 0;
                labels[i][1] = 1;
                labels[i][2] = 0;
            }
            if (data.get(i).variety.equals("Virginica")) {
                labels[i][0] = 0;
                labels[i][1] = 0;
                labels[i][2] = 1;
            }
        }
        return labels;
    }


    @Override
    public String getDescription() {
        return "Iris dataset";
    }
}
