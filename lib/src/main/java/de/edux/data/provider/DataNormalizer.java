package de.edux.data.provider;

import java.util.List;
import java.util.ArrayList;

public class DataNormalizer implements Normalizer {

    @Override
    public List<String[]> normalize(List<String[]> dataset) {
        if (dataset == null || dataset.isEmpty()) {
            return dataset;
        }

        int columnCount = dataset.get(0).length;

        double[] minValues = new double[columnCount];
        double[] maxValues = new double[columnCount];
        boolean[] isNumericColumn = new boolean[columnCount];

        for (int i = 0; i < columnCount; i++) {
            minValues[i] = Double.MAX_VALUE;
            maxValues[i] = -Double.MAX_VALUE;
            isNumericColumn[i] = true;
        }

        for (String[] row : dataset) {
            for (int colIndex = 0; colIndex < columnCount; colIndex++) {
                try {
                    double numValue = Double.parseDouble(row[colIndex]);

                    if (numValue < minValues[colIndex]) {
                        minValues[colIndex] = numValue;
                    }
                    if (numValue > maxValues[colIndex]) {
                        maxValues[colIndex] = numValue;
                    }
                } catch (NumberFormatException e) {
                    isNumericColumn[colIndex] = false;
                }
            }
        }

        for (String[] row : dataset) {
            for (int colIndex = 0; colIndex < columnCount; colIndex++) {
                if (isNumericColumn[colIndex]) {
                    double numValue = Double.parseDouble(row[colIndex]);
                    double range = maxValues[colIndex] - minValues[colIndex];

                    if (range != 0.0) {
                        double normalized = (numValue - minValues[colIndex]) / range;
                        row[colIndex] = String.valueOf(normalized);
                    } else {
                        row[colIndex] = "0";
                    }
                }
            }
        }

        return dataset;
    }

}
