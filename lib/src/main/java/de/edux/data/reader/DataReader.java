package de.edux.data.reader;

import java.io.File;
import java.util.List;

public interface DataReader {
    List<String[]> readFile(File file, char separator);
}
