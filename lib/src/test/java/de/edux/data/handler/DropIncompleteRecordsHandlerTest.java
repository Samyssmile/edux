package de.edux.data.handler;

import de.edux.data.provider.SeabornDataProcessor;
import de.edux.data.provider.SeabornProvider;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URL;
import java.util.Optional;

class DropIncompleteRecordsHandlerTest {
    private static final boolean SHUFFLE = true;
    private static final boolean SKIP_HEADLINE = true;
    private static final EIncompleteRecordsHandlerStrategy INCOMPLETE_RECORD_HANDLER_STRATEGY = EIncompleteRecordsHandlerStrategy.DROP_RECORDS;
    private static final double TRAIN_TEST_SPLIT_RATIO = 0.7;
    private static final String CSV_FILE_PATH = "testdatasets/seaborn-penguins/penguins.csv";
    private SeabornProvider seabornProvider;

    @Test
    void shouldReturnColumnData() {
        URL url = DropIncompleteRecordsHandlerTest.class.getClassLoader().getResource(CSV_FILE_PATH);
        if (url == null) {
            throw new IllegalStateException("Cannot find file: " + CSV_FILE_PATH);
        }
        File csvFile = new File(url.getPath());
        var seabornDataProcessor = new SeabornDataProcessor();
        var dataset = seabornDataProcessor.loadDataSetFromCSV(csvFile, ',', true, true, INCOMPLETE_RECORD_HANDLER_STRATEGY);
        seabornDataProcessor.normalize(dataset);
        Optional<Integer> indexOfSpecies =  seabornDataProcessor.getIndexOfColumn("species");
        String[] speciesData = seabornDataProcessor.getColumnDataOf("species");

        assert indexOfSpecies.isPresent();
        assert speciesData.length > 0;
    }
}