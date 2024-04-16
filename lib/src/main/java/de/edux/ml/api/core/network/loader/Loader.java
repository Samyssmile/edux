package de.edux.ml.api.core.network.loader;

public interface Loader {
  MetaData open();

  void close();

  MetaData getMetaData();

  BatchData readBatch();

  default void reset() {}
}
