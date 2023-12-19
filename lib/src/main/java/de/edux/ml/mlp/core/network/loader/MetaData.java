package de.edux.ml.mlp.core.network.loader;

public interface MetaData {
    int getNumberItems();

    void setNumberItems(int numberItems);

    int getInputSize();

    void setInputSize(int inputSize);

    int getExpectedSize();

    void setExpectedSize(int expectedSize);

    int getNumberBatches();

    void setNumberBatches(int numberBatches);

    int getTotalItemsRead();

    void setTotalItemsRead(int totalItemsRead);

    int getItemsRead();

    void setItemsRead(int itemsRead);
}
