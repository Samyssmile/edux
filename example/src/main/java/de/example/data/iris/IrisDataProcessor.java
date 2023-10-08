package de.example.data.iris;

import de.edux.data.provider.DataProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class IrisDataProcessor extends DataProcessor<Iris> {
    private static final Logger LOG = LoggerFactory.getLogger(IrisDataProcessor.class);
    private double[][] targets;

    @Override
    public void normalize(List<Iris> rowDataset) {
        double minSepalLength = rowDataset.stream().mapToDouble(iris -> iris.sepalLength).min().getAsDouble();
        double maxSepalLength = rowDataset.stream().mapToDouble(iris -> iris.sepalLength).max().getAsDouble();
        double minSepalWidth = rowDataset.stream().mapToDouble(iris -> iris.sepalWidth).min().getAsDouble();
        double maxSepalWidth = rowDataset.stream().mapToDouble(iris -> iris.sepalWidth).max().getAsDouble();
        double minPetalLength = rowDataset.stream().mapToDouble(iris -> iris.petalLength).min().getAsDouble();
        double maxPetalLength = rowDataset.stream().mapToDouble(iris -> iris.petalLength).max().getAsDouble();
        double minPetalWidth = rowDataset.stream().mapToDouble(iris -> iris.petalWidth).min().getAsDouble();
        double maxPetalWidth = rowDataset.stream().mapToDouble(iris -> iris.petalWidth).max().getAsDouble();

        for (Iris iris : rowDataset) {
            iris.sepalLength = (iris.sepalLength - minSepalLength) / (maxSepalLength - minSepalLength);
            iris.sepalWidth = (iris.sepalWidth - minSepalWidth) / (maxSepalWidth - minSepalWidth);
            iris.petalLength = (iris.petalLength - minPetalLength) / (maxPetalLength - minPetalLength);
            iris.petalWidth = (iris.petalWidth - minPetalWidth) / (maxPetalWidth - minPetalWidth);
        }
    }

    @Override
    public Iris mapToDataRecord(String[] csvLine) {
        return new Iris(
                Double.parseDouble(csvLine[0]),
                Double.parseDouble(csvLine[1]),
                Double.parseDouble(csvLine[2]),
                Double.parseDouble(csvLine[3]),
                csvLine[4]
        );
    }

    @Override
    public double[][] getInputs(List<Iris> dataset) {
        double[][] inputs = new double[dataset.size()][4];
        for (int i = 0; i < dataset.size(); i++) {
            inputs[i][0] = dataset.get(i).sepalLength;
            inputs[i][1] = dataset.get(i).sepalWidth;
            inputs[i][2] = dataset.get(i).petalLength;
            inputs[i][3] = dataset.get(i).petalWidth;
        }
        return inputs;
    }

    @Override
    public double[][] getTargets(List<Iris> dataset) {
        targets = new double[dataset.size()][3];
        for (int i = 0; i < dataset.size(); i++) {
            switch (dataset.get(i).variety) {
                case "Setosa":
                    targets[i][0] = 1;
                    break;
                case "Versicolor":
                    targets[i][1] = 1;
                    break;
                case "Virginica":
                    targets[i][2] = 1;
                    break;
            }
        }
        return targets;
    }

    @Override
    public double[][] getTrainFeatures() {
        return featuresOf(getSplitedDataset().trainData());
    }

    @Override
    public double[][] getTrainLabels() {
        return labelsOf(getSplitedDataset().trainData());
    }

    @Override
    public double[][] getTestLabels() {
        return labelsOf(getSplitedDataset().testData());
    }

    @Override
    public double[][] getTestFeatures() {
        return featuresOf(getSplitedDataset().testData());
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
    public String getDatasetDescription() {
        return "Iris dataset";
    }


}
