package de.edux.ml.mlp.core.network.loader.mnist;

import de.edux.ml.mlp.core.network.loader.BatchData;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.exceptions.LoaderException;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class MnistLoader implements Loader {
  private String imageFileName;
  private String labelFileName;
  private int batchSize;
  private DataInputStream imageInputStream;
  private DataInputStream labelInputStream;
  private MnistMetaData metaData;
  private Lock readLock = new ReentrantLock();

  public MnistLoader(String imageFileName, String labelFileName, int batchSize) {
    this.imageFileName = imageFileName;
    this.labelFileName = labelFileName;
    this.batchSize = batchSize;
  }

  @Override
  public MetaData open() {
    imageInputStream = getImageInputStream(imageFileName);
    labelInputStream = getImageInputStream(labelFileName);
    return readMetaData();
  }

  private DataInputStream getImageInputStream(String filename) {
    try {
      return new DataInputStream(new FileInputStream(filename));
    } catch (Exception e) {
      e.printStackTrace();
      throw new LoaderException(" Error opening file " + filename);
    }
  }

  @Override
  public void close() {
    metaData = null;
    try {
      imageInputStream.close();
      labelInputStream.close();

    } catch (IOException e) {
      throw new LoaderException("Error closing file " + imageFileName);
    }
  }

  @Override
  public MetaData getMetaData() {
    return metaData;
  }

  @Override
  public BatchData readBatch() {
    readLock.lock();
    MnistBatchData batchData;
    try {
      batchData = new MnistBatchData();
      int inputItemsRead = readInputBatch(batchData);
      int expectedItemsRead = readExpectedBatch(batchData);

      if (inputItemsRead != expectedItemsRead) {
        throw new LoaderException(
            "Number of input items read does not match number of expected items read");
      }
      metaData.setItemsRead(inputItemsRead);
    } finally {
      readLock.unlock();
    }
    return batchData;
  }

  private int readExpectedBatch(MnistBatchData batchData) {
    try {
      var totalItemsRead = metaData.getTotalItemsRead();
      var numberItems = metaData.getNumberItems();
      var numberToRead = Math.min(batchSize, numberItems - totalItemsRead);

      var labelData = new byte[numberToRead];
      var expectedSize = metaData.getNumberOfClasses();
      var numberRead = labelInputStream.read(labelData, 0, numberToRead);

      if (numberRead != numberToRead) {
        throw new LoaderException("Error reading expected data from file " + labelFileName);
      }

      double[] data = new double[numberToRead * expectedSize];
      for (int i = 0; i < numberToRead; i++) {
        byte label = labelData[i];
        data[i * expectedSize + label] = 1.0;
      }
      batchData.setExpectedBatch(data);
      return numberToRead;
    } catch (IOException e) {
      throw new LoaderException("Error reading input data from file " + imageFileName);
    }
  }

  private int readInputBatch(MnistBatchData batchData) {
    var totalItemsRead = metaData.getTotalItemsRead();
    var numberItems = metaData.getNumberItems();
    var numberToRead = Math.min(batchSize, numberItems - totalItemsRead);

    var inputSize = metaData.getInputSize();
    var numberBytesToRead = numberToRead * inputSize;

    byte[] imageData = new byte[numberBytesToRead];

    try {
      var numberRead = imageInputStream.read(imageData, 0, numberBytesToRead);
      if (numberRead != numberBytesToRead) {
        throw new LoaderException("Error reading input data from file " + imageFileName);
      }
      double[] data = new double[numberBytesToRead];

      for (int i = 0; i < numberBytesToRead; i++) {
        data[i] = (imageData[i] & 0xFF) / 256.0;
      }
      batchData.setInputBatch(data);
      return numberToRead;
    } catch (IOException e) {
      throw new LoaderException("Error reading input data from file " + imageFileName);
    }
  }

  private MetaData readMetaData() {
    int numberItems = 0;
    metaData = new MnistMetaData();
    try {
      int magicNumber = labelInputStream.readInt();
      if (magicNumber != 2049) {
        throw new LoaderException("Invalid magic number in file " + labelFileName);
      }

      numberItems = labelInputStream.readInt();
      metaData.setNumberItems(numberItems);

    } catch (IOException e) {
      throw new LoaderException("Error reading magic number from file " + labelFileName);
    }

    try {
      int magicNumber = imageInputStream.readInt();
      if (magicNumber != 2051) {
        throw new LoaderException("Invalid magic number in file " + labelFileName);
      }

      if (numberItems != imageInputStream.readInt()) {
        throw new LoaderException("Number of labels and images do not match");
      }

      int height = imageInputStream.readInt();
      int width = imageInputStream.readInt();
      metaData.setInputSize(height * width);
      metaData.setNumberOfClasses(10);
      metaData.setNumberBatches((int) Math.ceil(numberItems / batchSize));
      metaData.setHeight(height);
      metaData.setWidth(width);

    } catch (IOException e) {
      throw new LoaderException("Error reading magic number from file " + labelFileName);
    }
    return metaData;
  }
}
