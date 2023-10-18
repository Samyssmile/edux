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
    this.dataset.add(new String[] {"", "1", ""});
    this.dataset.add(new String[] {"C", "", "C"});
    this.dataset.add(new String[] {"D", "1", ""});
    this.dataset.add(new String[] {"E", "1", "E"});
    for (String[] data : dataset) {
      System.out.println(Arrays.toString(data));
    }

    List<String[]> cleanedDataset = incompleteRecordHandler.getCleanedDataset(dataset);
    System.out.println("----------------------------------------------------");
    for (String[] data : cleanedDataset) {
      System.out.println(Arrays.toString(data));
    }

    assertEquals(3, cleanedDataset.size());
  }

  @Test
  void fillWithAverageValues() {

    this.dataset.add(new String[] {"A", "1", "A"});
    this.dataset.add(new String[] {"", "1", ""});
    this.dataset.add(new String[] {"C", "", "C"});
    this.dataset.add(new String[] {"D", "1", ""});
    this.dataset.add(new String[] {"E", "1", "E"});
    for (String[] data : dataset) {
      System.out.println(Arrays.toString(data));
    }

    List<String[]> cleanedDataset = incompleteRecordHandler.getCleanedDataset(dataset);
    System.out.println("----------------------------------------------------");
    for (String[] data : cleanedDataset) {
      System.out.println(Arrays.toString(data));
    }

    assertEquals(1, Integer.valueOf(cleanedDataset.get(2)[1]));
  }
}
