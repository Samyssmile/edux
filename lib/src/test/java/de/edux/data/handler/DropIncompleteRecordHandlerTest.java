package de.edux.data.handler;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

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

    List<String[]> cleanedDataset = incompleteRecordHandler.getCleanedDataset(dataset);
    assertEquals(5, cleanedDataset.size());
  }

  @Test
  void testDropOneIncompleteResult() {

    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"A", "B", "C"});

    List<String[]> cleanedDataset = incompleteRecordHandler.getCleanedDataset(dataset);
    assertEquals(4, cleanedDataset.size());
  }

  @Test
  void testDropThreeIncompleteResults() {

    this.dataset.add(new String[] {"A", "B", "C"});
    this.dataset.add(new String[] {"", "B", "C"});
    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"A", "B", ""});
    this.dataset.add(new String[] {"A", "B", "C"});

    List<String[]> cleanedDataset = incompleteRecordHandler.getCleanedDataset(dataset);
    assertEquals(2, cleanedDataset.size());
  }

  @Test
  void testDropAllIncompleteResults() {

    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"", "B", "C"});
    this.dataset.add(new String[] {"A", "", "C"});
    this.dataset.add(new String[] {"A", "B", ""});
    this.dataset.add(new String[] {"A", "", "C"});

    List<String[]> cleanedDataset = incompleteRecordHandler.getCleanedDataset(dataset);
    assertEquals(0, cleanedDataset.size());
  }
}
