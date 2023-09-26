package de.edux.data.provider;

import de.edux.data.reader.CSVIDataReader;
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
class DataUtilTest {
    @InjectMocks
    private DataUtil<String> dataUtil = getDummyDataUtil();

    @Mock
    private CSVIDataReader csvDataReader;

    @BeforeEach
    void setUp() {
        dataUtil = getDummyDataUtil();
    }

    @Test
    void testSplitWithValidRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 0.6;

        List<List<String>> result = dataUtil.split(dataset, trainTestSplitRatio);

        assertEquals(3, result.get(0).size(), "Train dataset size should be 3");
        assertEquals(2, result.get(1).size(), "Test dataset size should be 2");
    }

    @Test
    void testSplitWithZeroRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 0.0;

        List<List<String>> result = dataUtil.split(dataset, trainTestSplitRatio);

        assertEquals(0, result.get(0).size(), "Train dataset size should be 0");
        assertEquals(5, result.get(1).size(), "Test dataset size should be 5");
    }

    @Test
    void testSplitWithFullRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 1.0;

        List<List<String>> result = dataUtil.split(dataset, trainTestSplitRatio);

        assertEquals(5, result.get(0).size(), "Train dataset size should be 5");
        assertEquals(0, result.get(1).size(), "Test dataset size should be 0");
    }

    @Test
    void testSplitWithInvalidNegativeRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = -0.1;

        assertThrows(IllegalArgumentException.class, () -> dataUtil.split(dataset, trainTestSplitRatio));
    }

    @Test
    void testSplitWithInvalidAboveOneRatio() {
        List<String> dataset = Arrays.asList("A", "B", "C", "D", "E");
        double trainTestSplitRatio = 1.1;

        assertThrows(IllegalArgumentException.class, () -> dataUtil.split(dataset, trainTestSplitRatio));
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

        List<String> result = dataUtil.loadTDataSet(dummyFile, separator, false, false, false);

        assertEquals(2, result.size(), "Dataset size should be 2");
    }

    private DataUtil<String> getDummyDataUtil() {
        return new DataUtil<>(csvDataReader) {

            @Override
            public void normalize(List<String> dataset) {
                // Mock normalize for the sake of testing
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
        };
    }

}
