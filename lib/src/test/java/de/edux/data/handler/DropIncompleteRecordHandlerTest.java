package de.edux.data.handler;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DropIncompleteRecordHandlerTest {
  private List<String[]> dataset;

  private IIncompleteRecordsHandler incompleteRecordHandler;

  @BeforeEach
  void initializeList() {
    dataset = new ArrayList<>();
    incompleteRecordHandler = EIncompleteRecordsHandlerStrategy.DROP_RECORDS.getHandler();
  }

  @Test
  void testDropZeroIncompleteResults() {

    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});

    assertEquals(5, incompleteRecordHandler.getCleanedDataset(dataset).size());
  }

  @Test
  void testDropOneIncompleteResult() {

    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});

    assertEquals(4, incompleteRecordHandler.getCleanedDataset(dataset).size());
  }

  @Test
  void testDropTwoIncompleteResult() {

    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});

    assertEquals(3, incompleteRecordHandler.getCleanedDataset(dataset).size());
  }

  @Test
  void testThrowRuntimeExceptionForDroppingMoreThanHalfOfOriginalDataset() {

    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"", "B", "C"});
    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"A", "B", ""});
    this.dataset.add(new String[] {"A", "B", "C"});

    assertThrows(RuntimeException.class, () -> incompleteRecordHandler.getCleanedDataset(dataset));
  }
}
