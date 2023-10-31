package de.edux.data.provider;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyChar;
import static org.mockito.Mockito.when;

import de.edux.data.reader.IDataReader;
import de.edux.functions.imputation.ImputationStrategy;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class DataProcessorTest {
    private static final boolean SKIP_HEAD = true;
    @Mock
    IDataReader dataReader;
    List<String[]> dummyDatasetForImputationTest;
    private List<String[]> dummyDataset;
    private DataProcessor dataProcessor;

    @BeforeEach
    void setUp() {
        dummyDataset = new ArrayList<>();
        dummyDataset.add(new String[]{"col1", "col2", "Name", "col4", "col5"});
        dummyDataset.add(new String[]{"1", "2", "3", "Anna", "5"});
        dummyDataset.add(new String[]{"6", "7", "8", "Nina", "10"});
        dummyDataset.add(new String[]{"11", "12", "13", "Johanna", "15"});
        dummyDataset.add(new String[]{"16", "17", "18", "Isabela", "20"});
        when(dataReader.readFile(any(), anyChar())).thenReturn(dummyDataset);

        dummyDatasetForImputationTest = new ArrayList<>();
        dummyDatasetForImputationTest.add(new String[] {"Fruit", "Quantity", "Price"});
        dummyDatasetForImputationTest.add(new String[] {"Apple", "", "8"});
        dummyDatasetForImputationTest.add(new String[] {"Apple", "2", "9"});
        dummyDatasetForImputationTest.add(new String[] {"", "3", "10"});
        dummyDatasetForImputationTest.add(new String[] {"Peach", "3", ""});
        dummyDatasetForImputationTest.add(new String[] {"Kiwi", "5", ""});
        dummyDatasetForImputationTest.add(new String[] {"", "3", "11"});
        dummyDatasetForImputationTest.add(new String[] {"Banana", "7", "12"});

        dataProcessor = new DataProcessor(dataReader);
    }

    @Test
    void shouldSkipHead() {
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', SKIP_HEAD, new int[]{0, 1, 2, 4}, 3);
        assertEquals(4, dataProcessor.getDataset().size(), "Number of rows does not match.");
    }

    @Test
    void shouldNotSkipHead() {
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', false, new int[]{0, 1, 2, 4}, 3);
        assertEquals(5, dataProcessor.getDataset().size(), "Number of rows does not match.");
    }


    @Test
    void getTargets() {
        when(dataReader.readFile(any(), anyChar())).thenReturn(dummyDataset);
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', SKIP_HEAD, new int[]{0, 1, 2, 4}, 3)
                        .split(0.5);
        dummyDataset.add(new String[]{"21", "22", "23", "Isabela", "25"});

        double[][] targets = dataProcessor.getTargets(dummyDataset, 3);
        double[][] expectedTargets = {
                {1.0, 0.0, 0.0, 0.0}, // Anna
                {0.0, 1.0, 0.0, 0.0}, // Nina
                {0.0, 0.0, 1.0, 0.0}, // Johanna
                {0.0, 0.0, 0.0, 1.0}, // Isabela
                {0.0, 0.0, 0.0, 1.0}  // Isabela
        };

        for (int i = 0; i < expectedTargets.length; i++) {
            assertArrayEquals(expectedTargets[i], targets[i], "Die Zielzeile " + i + " stimmt nicht überein.");
        }

        Map<String, Integer> classMap = dataProcessor.getClassMap();
        Map<String, Integer> expectedClassMap = Map.of(
                "Anna", 0,
                "Nina", 1,
                "Johanna", 2,
                "Isabela", 3);

        assertEquals(expectedClassMap, classMap, "Die Klassen stimmen nicht überein.");
    }

    @Test
    void getInputs() {
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', SKIP_HEAD, new int[]{0, 1, 2, 4}, 3)
                .split(0.5);
        double[][] inputs = dataProcessor.getInputs(dummyDataset, new int[]{0, 1, 2, 4});

        double[][] expectedInputs = {
                {1.0, 2.0, 3.0, 5.0},
                {6.0, 7.0, 8.0, 10.0},
                {11.0, 12.0, 13.0, 15.0},
                {16.0, 17.0, 18.0, 20.0}
        };

        assertEquals(expectedInputs.length, inputs.length, "Die Anzahl der Zeilen stimmt nicht überein.");

        for (int i = 0; i < expectedInputs.length; i++) {
            assertArrayEquals(expectedInputs[i], inputs[i], "Die Zeile " + i + " entspricht nicht den erwarteten Werten.");
        }
    }

    private List<String[]> duplicateList(List<String[]> list) {
        List<String[]> duplicate = new ArrayList<>();
        for (String[] row : list) {
            duplicate.add(row.clone());
        }
        return duplicate;
    }

    @Test
    void shouldNormalize() {
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', SKIP_HEAD, new int[]{0, 1, 2, 4}, 3)
                .split(0.5);
        List<String[]> normalizedDataset = dataProcessor.normalize().getDataset();

        String[][] expectedNormalizedValues = {
                {"0.0", "0.0", "0.0", "Anna", "0.0"},
                {"0.3333333333333333", "0.3333333333333333", "0.3333333333333333", "Nina", "0.3333333333333333"},
                {"0.6666666666666666", "0.6666666666666666", "0.6666666666666666", "Johanna", "0.6666666666666666"},
                {"1.0", "1.0", "1.0", "Isabela", "1.0"}
        };

        for (int i = 1; i < normalizedDataset.size(); i++) {
            String[] row = normalizedDataset.get(i);
            assertArrayEquals(expectedNormalizedValues[i], row, "Die Zeile " + i + " entspricht nicht den erwarteten normalisierten Werten.");
        }
    }

    @Test
    void shouldShuffle() {
        List<String[]> originalDataset = duplicateList(dummyDataset);
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', false, new int[]{0, 1, 2, 4}, 3)
                .split(0.5);
        List<String[]> shuffledDataset = dataProcessor.shuffle().getDataset();

        assertNotEquals(originalDataset, shuffledDataset, "Die Reihenfolge der Zeilen hat sich nicht geändert.");
    }


    @Test
    void shouldReturnTrainTestDataset() {
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', false, new int[]{0, 1, 2, 4}, 3);
        dataProcessor.split(0.5);

        int[] inputColumns = new int[]{0, 1, 2, 4};
        double[][] trainFeatures = dataProcessor.getTrainFeatures(inputColumns);
        double[][] testFeatures = dataProcessor.getTestFeatures(inputColumns);

        double[][] trainLabels = dataProcessor.getTrainLabels(3);
        double[][] testLabels = dataProcessor.getTestLabels(3);

    }

    @Test
    void shouldPerformImputationOnDataset() {
        when(dataReader.readFile(any(), anyChar())).thenReturn(dummyDatasetForImputationTest);
        dataProcessor.loadDataSetFromCSV(new File("mockpathhere"), ',', SKIP_HEAD, new int[]{0, 1}, 2);

        ImputationStrategy modeImputter = ImputationStrategy.MODE;
        ImputationStrategy averageImputter = ImputationStrategy.AVERAGE;

        dataProcessor.imputation("Fruit",modeImputter);
        dataProcessor.imputation("Quantity",modeImputter);
        dataProcessor.imputation("Price",averageImputter);
        var imputtedDataset = dataProcessor.getDataset();

        assertAll(
            () -> assertArrayEquals(new String[] {"Apple", "3", "8"}, imputtedDataset.get(0)),
            () -> assertArrayEquals(new String[] {"Apple", "2", "9"}, imputtedDataset.get(1)),
            () -> assertArrayEquals(new String[] {"Apple", "3", "10"}, imputtedDataset.get(2)),
            () -> assertArrayEquals(new String[] {"Peach", "3", "10.0"}, imputtedDataset.get(3)),
            () -> assertArrayEquals(new String[] {"Kiwi", "5", "10.0"}, imputtedDataset.get(4)),
            () -> assertArrayEquals(new String[] {"Apple", "3", "11"}, imputtedDataset.get(5)),
            () -> assertArrayEquals(new String[] {"Banana", "7", "12"}, imputtedDataset.get(6))
        );
  }
}