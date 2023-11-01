package de.edux.data.provider;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.List;
import org.junit.jupiter.api.Test;

public class DataNormalizerTest {

  private DataNormalizer normalizer = new DataNormalizer();

  @Test
  public void whenDatasetIsNull_returnSameDataset() {
    List<String[]> dataset = null;
    assertSame(dataset, normalizer.normalize(dataset));
  }

  @Test
  public void whenDatasetIsEmpty_returnSameDataset() {
    List<String[]> dataset = List.of();
    assertSame(dataset, normalizer.normalize(dataset));
  }

  @Test
  public void whenAllColumnsAreNumeric_normalizeValues() {
    List<String[]> dataset = List.of(new String[] {"1", "2"}, new String[] {"3", "4"});

    List<String[]> normalized = normalizer.normalize(dataset);

    assertEquals("0.0", normalized.get(0)[0]);
    assertEquals("0.0", normalized.get(0)[1]);
    assertEquals("1.0", normalized.get(1)[0]);
    assertEquals("1.0", normalized.get(1)[1]);
  }

  @Test
  public void whenColumnsAreNonNumeric_ignoreNonNumericColumns() {
    List<String[]> dataset = List.of(new String[] {"1", "a"}, new String[] {"3", "b"});

    List<String[]> normalized = normalizer.normalize(dataset);

    assertEquals("0.0", normalized.get(0)[0]);
    assertEquals("a", normalized.get(0)[1]);
    assertEquals("1.0", normalized.get(1)[0]);
    assertEquals("b", normalized.get(1)[1]);
  }

  @Test
  public void whenColumnValuesAreEqual_returnZero() {
    List<String[]> dataset = List.of(new String[] {"2", "2"}, new String[] {"2", "2"});

    List<String[]> normalized = normalizer.normalize(dataset);

    assertEquals("0", normalized.get(0)[0]);
    assertEquals("0", normalized.get(0)[1]);
    assertEquals("0", normalized.get(1)[0]);
    assertEquals("0", normalized.get(1)[1]);
  }
}
