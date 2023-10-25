package de.edux.data.reader;

import java.io.File;
import java.util.List;

public interface IDataReader {
    List<String[]> readFile(File file, char separator);
}
