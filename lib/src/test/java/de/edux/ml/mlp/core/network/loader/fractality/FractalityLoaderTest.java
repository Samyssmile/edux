package de.edux.ml.mlp.core.network.loader.fractality;

import static org.junit.jupiter.api.Assertions.*;

import de.edux.ml.mlp.core.network.loader.MetaData;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class FractalityLoaderTest {

  private static FractalityLoader fractalityLoader;

  @BeforeAll
  static void setUp() {
    fractalityLoader =
        new FractalityLoader(
            "src/test/resources/fractality/test/class",
            "src/test/resources/fractality/test/images.csv",
            5,
            256,
            256);
  }

  @Test
  void shouldLoadCSVContent() {
    assertEquals(120, fractalityLoader.getCsvContent().size());
    assertTrue(fractalityLoader.getCsvContent().values().stream().allMatch(s -> s != null));
  }

  @Test
  void shouldReadMetaData() {
    MetaData metaData = fractalityLoader.open();
    assertEquals(120, metaData.getNumberItems());
    assertNotNull(fractalityLoader.getMetaData());
    assertEquals(6, metaData.getNumberOfClasses());
    assertEquals(24, metaData.getNumberBatches());
    assertEquals(0, metaData.getTotalItemsRead());
    assertEquals(5, metaData.getBatchLength());
  }

  @Test
  void shouldReadBatches() {
    MetaData metaData = fractalityLoader.open();
    for (int i = 0; i < metaData.getNumberBatches(); i++) {
      var batchData = fractalityLoader.readBatch();
      assertNotNull(batchData);
    }
  }
}
