package de.edux.ml.cnn.data;

import java.util.Iterator;

public interface DataLoader extends Iterator<Batch> {
    void shuffle();
    void reset();
    int size();
}