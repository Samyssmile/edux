package de.edux.data.handler;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AverageFillIncompleteRecordHandlerTest {

  private List<String[]> dataset;

  private IIncompleteRecordsHandler incompleteRecordHandler;

  @BeforeEach
  void initializeList() {
    dataset = new ArrayList<>();
    incompleteRecordHandler =
        EIncompleteRecordsHandlerStrategy.FILL_RECORDS_WITH_AVERAGE.getHandler();
  }

  @Test
  void dropRecordsWithIncompleteCategoricalFeature() {

    this.dataset.add(new String[] {"A", "1", "A"});
    this.dataset.add(new String[] {"", "2", ""});
    this.dataset.add(new String[] {"C", "", "C"});
    this.dataset.add(new String[] {"D", "3", ""});
    this.dataset.add(new String[] {"E", "4", "E"});

    assertAll(
        () -> assertEquals(3, incompleteRecordHandler.getCleanedDataset(dataset).size()),
        () ->
            assertEquals(
                2.5, Double.valueOf(incompleteRecordHandler.getCleanedDataset(dataset).get(1)[1])));
  }

  @Test
  void testThrowRuntimeExceptionForDroppingMoreThanHalfOfOriginalDataset() {

    this.dataset.add(new String[] {"", "1", "A"});
    this.dataset.add(new String[] {"B", "2", "B"});
    this.dataset.add(new String[] {"C", "3", "C"});
    this.dataset.add(new String[] {"D", "4", ""});
    this.dataset.add(new String[] {"", "5", "E"});

    assertThrows(RuntimeException.class, () -> incompleteRecordHandler.getCleanedDataset(dataset));
  }

  @Test
  void testThrowRuntimeExceptionForZeroValidNumericalFeatures() {

    this.dataset.add(new String[] {"A", "", "A"});
    this.dataset.add(new String[] {"B", "", "B"});
    this.dataset.add(new String[] {"C", "1", "C"});
    this.dataset.add(new String[] {"D", "", "D"});
    this.dataset.add(new String[] {"E", "", "E"});

    assertThrows(RuntimeException.class, () -> incompleteRecordHandler.getCleanedDataset(dataset));
  }

  @Test
  void testThrowRuntimeExceptionForAtLeastOneFullValidRecord() {

    this.dataset.add(new String[] {"", "1", "A"});
    this.dataset.add(new String[] {"B", "2", ""});
    this.dataset.add(new String[] {"", "", "C"});
    this.dataset.add(new String[] {"D", "3", ""});
    this.dataset.add(new String[] {"", "4", "E"});

    assertThrows(RuntimeException.class, () -> incompleteRecordHandler.getCleanedDataset(dataset));
  }
}
