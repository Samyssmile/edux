package de.edux.ml.mlp.core.network.loader.csv;

import de.edux.ml.mlp.core.network.loader.BatchData;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;

import java.io.File;

public class CSVDataLoader implements Loader {
    public CSVDataLoader(File csvFile, int batchSize) {
    }

    @Override
    public MetaData open() {
        return null;
    }

    @Override
    public void close() {

    }

    @Override
    public MetaData getMetaData() {
        return null;
    }

    @Override
    public BatchData readBatch() {
        return null;
    }
}
