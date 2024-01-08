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
            "src/test/resources/fractality/small_test/class",
            "src/test/resources/fractality/small_test/images.csv",
            5,
            64,
            64);
  }

  @Test
  void shouldLoadCSVContent() {
    assertEquals(59, fractalityLoader.getCsvContent().size());
    assertTrue(fractalityLoader.getCsvContent().values().stream().allMatch(s -> s != null));
  }

  @Test
  void shouldReadMetaData() {
    MetaData metaData = fractalityLoader.open();
    assertEquals(59, metaData.getNumberItems());
    assertNotNull(fractalityLoader.getMetaData());
    assertEquals(6, metaData.getNumberOfClasses());
    assertEquals(11, metaData.getNumberBatches());
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
