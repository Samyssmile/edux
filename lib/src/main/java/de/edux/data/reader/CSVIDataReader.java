package de.edux.data.reader;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvException;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

public class CSVIDataReader implements IDataReader {

    public List<String[]> readFile(File file, char separator) {
        CSVParser customCSVParser = new CSVParserBuilder().withSeparator(separator).build(); // custom separator
        List<String[]> result;
        try(CSVReader reader = new CSVReaderBuilder(
                new FileReader(file))
                .withCSVParser(customCSVParser)
                .withSkipLines(1)
                .build()){
            result = reader.readAll();
        } catch (CsvException | IOException e) {
            throw new RuntimeException(e);
        }
        return result;
    }

}
