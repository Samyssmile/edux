package de.edux.data.provider;

import java.util.List;

public interface Normalizer {
    List<String[]> normalize(List<String[]> dataset);
}
