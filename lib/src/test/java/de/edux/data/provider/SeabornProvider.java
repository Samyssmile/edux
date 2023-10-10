package de.edux.data.provider;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class SeabornProvider implements IDataProvider<Penguin> {
    private static final Logger LOG = LoggerFactory.getLogger(SeabornProvider.class);
    private final List<Penguin> dataset;
    private final List<Penguin> trainingData;
    private final List<Penguin> testData;

    public SeabornProvider(List<Penguin> dataset, List<Penguin> trainingData, List<Penguin> testData) {
        this.dataset = dataset;
        this.trainingData = trainingData;
        this.testData = testData;
    }

    @Override
    public List<Penguin> getTrainData() {
        return trainingData;
    }

    @Override
    public List<Penguin> getTestData() {
        return testData;
    }

    @Override
    public void printStatistics() {
        LOG.info("========================= Data Statistic ==================");
        LOG.info("Dataset: Seaborn Penguins");
        LOG.info("Description: " + getDescription());
        LOG.info("Total dataset size: " + dataset.size());
        LOG.info("Training dataset size: " + trainingData.size());
        LOG.info("Test data set size: " + testData.size());
        LOG.info("Classes: " + getTrainLabels()[0].length);
        LOG.info("===========================================================");
    }

    @Override
    public Penguin getRandom(boolean equalDistribution) {
        return dataset.get((int) (Math.random() * dataset.size()));
    }

    @Override
    public double[][] getTrainFeatures() {
        return featuresOf(trainingData);
    }

    private double[][] featuresOf(List<Penguin> data) {
        double[][] features = new double[data.size()][4];

        for (int i = 0; i < data.size(); i++) {
            Penguin p = data.get(i);
            features[i][0] = p.billLengthMm();
            features[i][1] = p.billDepthMm();
            features[i][2] = p.flipperLengthMm();
            features[i][3] = p.bodyMassG();
        }

        return features;
    }


    @Override
    public double[][] getTrainLabels() {
        return labelsOf(trainingData);
    }

    private double[][] labelsOf(List<Penguin> data) {
        double[][] labels = new double[data.size()][3];

        for (int i = 0; i < data.size(); i++) {
            Penguin p = data.get(i);
            switch (p.species().toLowerCase()) {
                case "adelie":
                    labels[i] = new double[]{1.0, 0.0, 0.0};
                    break;
                case "chinstrap":
                    labels[i] = new double[]{0.0, 1.0, 0.0};
                    break;
                case "gentoo":
                    labels[i] = new double[]{0.0, 0.0, 1.0};
                    break;
                default:
                    throw new IllegalArgumentException("Unbekannte Pinguinart: " + p.species());
            }
        }

        return labels;
    }

    @Override
    public double[][] getTestFeatures() {
        return featuresOf(testData);
    }

    @Override
    public double[][] getTestLabels() {
        return labelsOf(testData);
    }

    @Override
    public String getDescription() {
        return "The Seaborn Penguin dataset comprises measurements and species classifications for penguins collected from three islands in the Palmer Archipelago, Antarctica.";
    }
}
