package de.nexent.edux.data.reader;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvException;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

public class CSVDataReader implements DataReader {

    public List<String[]> readFile(File file, char separator) {
        CSVParser csvParser = new CSVParserBuilder().withSeparator(separator).build(); // custom separator
        List<String[]> result = null;
        try(CSVReader reader = new CSVReaderBuilder(
                new FileReader(file))
                .withCSVParser(csvParser)   // custom CSV parser
                .withSkipLines(1)           // skip the first line, header info
                .build()){
            result = reader.readAll();
        } catch (CsvException | IOException e) {
            throw new RuntimeException(e);
        }
        return result;
    }

}
