package de.example.data;

import de.edux.data.reader.CSVDataReader;
import de.edux.data.reader.DataReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class IrisDataUtil {
    private static final Logger logger = LoggerFactory.getLogger(IrisDataUtil.class);
    public static List<Iris> loadIrisDataSet(boolean normalize, boolean shuffle) {
        DataReader dataReader = new CSVDataReader();
        File csvFile = new File("example"+ File.separator + "datasets"+ File.separator +  "iris" + File.separator + "iris.csv");
        List<String[]> csvLines = dataReader.readFile(csvFile, ',');
        List<Iris> unmodifiableDataset = csvLines.stream().map(IrisDataUtil::mapToIris).toList();
        List<Iris> dataset = new ArrayList<>(unmodifiableDataset);
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

    private static void normalize(List<Iris> rowDataset) {
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

    private static Iris mapToIris(String[] csvLine) {
        return new Iris(
                Double.parseDouble(csvLine[0]),
                Double.parseDouble(csvLine[1]),
                Double.parseDouble(csvLine[2]),
                Double.parseDouble(csvLine[3]),
                csvLine[4]
        );
    }

    public static double[][] getInputs(List<Iris> dataset) {
        double[][] inputs = new double[dataset.size()][4];
        for (int i = 0; i < dataset.size(); i++) {
            inputs[i][0] = dataset.get(i).sepalLength;
            inputs[i][1] = dataset.get(i).sepalWidth;
            inputs[i][2] = dataset.get(i).petalLength;
            inputs[i][3] = dataset.get(i).petalWidth;
        }
        return inputs;
    }

    public static double[][] getTargets(List<Iris> dataset) {
        double[][] targets = new double[dataset.size()][3];
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

    public static List<List<Iris>> split(List<Iris> dataset, double v) {
        int splitIndex = (int) (dataset.size() * v);
        return List.of(dataset.subList(0, splitIndex), dataset.subList(splitIndex, dataset.size()));
    }
}
