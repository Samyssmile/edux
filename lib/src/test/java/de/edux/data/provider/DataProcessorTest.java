package de.edux.data.provider;

import de.edux.data.handler.EIncompleteRecordsHandlerStrategy;
import de.edux.data.reader.CSVIDataReader;
import de.edux.ml.nn.network.api.Dataset;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyChar;
import static org.mockito.Mockito.when;


@ExtendWith(MockitoExtension.class)
class DataProcessorTest {
    @InjectMocks
    private DataProcessor<String> dataProcessor = getDummyDataUtil();

    @Mock
    private CSVIDataReader csvDataReader;

    @BeforeEach
    void setUp() {
        dataProcessor = getDummyDataUtil();
    }

    @Test
    void testSplitWithValidRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 0.6;

        Dataset<String> result = dataProcessor.split(dataset, trainTestSplitRatio);

        assertEquals(3, result.trainData().size(), "Train dataset size should be 3");
        assertEquals(2, result.testData().size(), "Test dataset size should be 2");
    }

    @Test
    void testSplitWithZeroRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 0.0;

        Dataset<String> result = dataProcessor.split(dataset, trainTestSplitRatio);

        assertEquals(0, result.trainData().size(), "Train dataset size should be 0");
        assertEquals(5, result.testData().size(), "Test dataset size should be 5");
    }

    @Test
    void testSplitWithFullRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 1.0;

        Dataset<String> result = dataProcessor.split(dataset, trainTestSplitRatio);

        assertEquals(5, result.trainData().size(), "Train dataset size should be 5");
        assertEquals(0, result.testData().size(), "Test dataset size should be 0");
    }

    @Test
    void testSplitWithInvalidNegativeRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = -0.1;

        assertThrows(IllegalArgumentException.class, () -> dataProcessor.split(dataset, trainTestSplitRatio));
    }

    @Test
    void testSplitWithInvalidAboveOneRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 1.1;

        assertThrows(IllegalArgumentException.class, () -> dataProcessor.split(dataset, trainTestSplitRatio));
    }

    @Test
    void testLoadTDataSetWithoutNormalizationAndShuffling() {
        File dummyFile = new File("dummy.csv");
        char separator = ',';
        String[] csvFirstLine = {"A", "B", "C", "D", "E"};
        String[] csvSecondLine = {"F", "G", "H", "I", "J"};
        List<String[]> csvLine = new ArrayList<>();
        csvLine.add(csvFirstLine);
        csvLine.add(csvSecondLine);

        when(csvDataReader.readFile(any(), anyChar())).thenReturn(csvLine);

        List<String> result = dataProcessor.loadDataSetFromCSV(dummyFile, separator, false, false, EIncompleteRecordsHandlerStrategy.DO_NOT_HANDLE);
        assertEquals(2, result.size(), "Dataset size should be 2");
    }

    private DataProcessor<String> getDummyDataUtil() {
        return new DataProcessor<>(csvDataReader) {

            @Override
            public void normalize(List<String> dataset) {
            }

            @Override
            public String mapToDataRecord(String[] csvLine) {
                return null;
            }

            @Override
            public double[][] getInputs(List<String> dataset) {
                return new double[0][];
            }

            @Override
            public double[][] getTargets(List<String> dataset) {
                return new double[0][];
            }

            @Override
            public String getDatasetDescription() {
                return null;
            }

            @Override
            public double[][] getTrainFeatures() {
                return new double[0][];
            }

            @Override
            public double[][] getTrainLabels() {
                return new double[0][];
            }

            @Override
            public double[][] getTestLabels() {
                return new double[0][];
            }

            @Override
            public double[][] getTestFeatures() {
                return new double[0][];
            }
        };
    }

}
