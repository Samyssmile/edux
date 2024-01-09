package de.edux.ml.mlp.core.network.loader.mnist;

import de.edux.ml.mlp.core.network.loader.AbstractMetaData;

public class MnistMetaData extends AbstractMetaData {
  private int width;
  private int height;

  public int getWidth() {
    return width;
  }

  public void setWidth(int width) {
    this.width = width;
  }

  public int getHeight() {
    return height;
  }

  public void setHeight(int height) {
    this.height = height;
  }

  @Override
  public void setItemsRead(int itemsRead) {
    super.setItemsRead(itemsRead);
    super.setTotalItemsRead(super.getTotalItemsRead() + itemsRead);
  }
}
