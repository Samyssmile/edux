package de.edux.core.network.loader;

import static org.junit.jupiter.api.Assertions.*;

import de.edux.ml.mlp.core.network.loader.BatchData;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.core.network.loader.test.TestLoader;
import de.edux.ml.mlp.core.tensor.Matrix;
import org.junit.jupiter.api.Test;

class LoaderTest {

    @Test
    void shouldOpen() {
        var batchSize = 33;
        Loader loader = new TestLoader(600, batchSize, 33);
        MetaData metaData = loader.open();

        int numberItems = metaData.getNumberItems();
        int lastBatchSize = numberItems % batchSize;
        int numberBatches = metaData.getNumberBatches();

        for (int i = 0; i <numberBatches; i++) {
            BatchData batchData = loader.readBatch();
            assertNotNull(batchData);

            int itemsRead = metaData.getItemsRead();

            int inputSize = metaData.getInputSize();
            int expectedSize = metaData.getExpectedSize();

            Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
            Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());

            System.out.println(input.sum());
            assertTrue(input.sum() != 0.0);

            assertTrue(expected.sum() == itemsRead);


            if (i == numberBatches -1){
                assertEquals(itemsRead, lastBatchSize);
            }else{
                assertEquals(itemsRead, batchSize);
            }
        }

    }

}