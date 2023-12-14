package de.edux.ml.mlp.core.network.loader;

public interface Loader {
    MetaData open();
    void close();

    MetaData getMetaData();
    BatchData readBatch();


}
