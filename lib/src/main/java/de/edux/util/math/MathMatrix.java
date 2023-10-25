package de.edux.util.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MathMatrix implements ConcurrentMatrixMultiplication {
    private static final Logger LOG = LoggerFactory.getLogger(MathMatrix.class);

    @Override
    public double[][] multiplyMatrices(double[][] a, double[][] b) throws IncompatibleDimensionsException {
        LOG.info("Multiplying matrices of size {}x{} and {}x{}", a.length, a[0].length, b.length, b[0].length);
        int aRows = a.length;
        int aCols = a[0].length;
        int bCols = b[0].length;

        if (aCols != b.length) {
            throw new IncompatibleDimensionsException("Cannot multiply matrices with incompatible dimensions");
        }

        double[][] result = new double[aRows][bCols];

        try(var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<Void>> futures = new ArrayList<>(aRows);

            for (int i = 0; i < aRows; i++) {
                final int rowIndex = i;
                futures.add(executor.submit(() -> {
                    for (int colIndex = 0; colIndex < bCols; colIndex++) {
                        result[rowIndex][colIndex] = multiplyMatrixRowByColumn(a, b, rowIndex, colIndex);
                    }
                    return null;
                }));
            }
            for (var future : futures) {
                future.get();
            }
        } catch (ExecutionException | InterruptedException e) {
            LOG.error("Error while multiplying matrices", e);
        }

        LOG.info("Finished multiplying matrices");
        return result;
    }

    private double multiplyMatrixRowByColumn(double[][] a, double[][] b, int row, int col) {
        double sum = 0;
        for (int i = 0; i < a[0].length; i++) {
            sum += a[row][i] * b[i][col];
        }
        return sum;
    }
}
