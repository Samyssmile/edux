package de.edux.ml.mlp.core.network.loader.fractality;

import de.edux.ml.mlp.core.network.loader.AbstractMetaData;

public class FractalityMetaData extends AbstractMetaData {
  @Override
  public void setItemsRead(int itemsRead) {
    super.setItemsRead(itemsRead);
    super.setTotalItemsRead(super.getTotalItemsRead() + itemsRead);
  }
}
