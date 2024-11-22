package de.edux.ml.mlp.core.network.loader;

public abstract class AbstractMetaData implements MetaData {
  private int numberItems;
  private int inputSize;
  private int numberOfClasses;
  private int numberBatches;
  private int totalItemsRead;
  private int itemsRead;
  private int batchLength;

  @Override
  public int getNumberItems() {
    return numberItems;
  }

  @Override
  public void setNumberItems(int numberItems) {
    this.numberItems = numberItems;
  }

  @Override
  public int getInputSize() {
    return inputSize;
  }

  @Override
  public void setInputSize(int inputSize) {
    this.inputSize = inputSize;
  }

  @Override
  public int getNumberOfClasses() {
    return numberOfClasses;
  }

  @Override
  public void setNumberOfClasses(int numberOfClasses) {
    this.numberOfClasses = numberOfClasses;
  }

  @Override
  public int getNumberBatches() {
    return numberBatches;
  }

  @Override
  public void setNumberBatches(int numberBatches) {
    this.numberBatches = numberBatches;
  }

  @Override
  public int getTotalItemsRead() {
    return totalItemsRead;
  }

  @Override
  public void setTotalItemsRead(int totalItemsRead) {
    this.totalItemsRead = totalItemsRead;
  }

  @Override
  public int getItemsRead() {
    return itemsRead;
  }

  @Override
  public void setItemsRead(int itemsRead) {
    this.itemsRead = itemsRead;
  }

  @Override
  public int getBatchLength() {
    return batchLength;
  }

  @Override
  public void setBatchLength(int batchLength) {
    this.batchLength = batchLength;
  }

  @Override
    public void printDatasetInformation() {
        System.out.println("Number of items: " + numberItems);
        System.out.println("Input size: " + inputSize);
        System.out.println("Number of classes: " + numberOfClasses);
        System.out.println("Number of batches: " + numberBatches);
        System.out.println("Total items read: " + totalItemsRead);
        System.out.println("Items read: " + itemsRead);
        System.out.println("Batch length: " + batchLength);
    }
}
