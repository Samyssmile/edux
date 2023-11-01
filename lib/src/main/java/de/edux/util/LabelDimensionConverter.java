package de.edux.util;

public class LabelDimensionConverter {
  public static int[] convert2DLabelArrayTo1DLabelArray(double[][] labels) {
    int[] decisionTreeTrainLabels = new int[labels.length];
    for (int i = 0; i < labels.length; i++) {
      for (int j = 0; j < labels[i].length; j++) {
        if (labels[i][j] == 1) {
          decisionTreeTrainLabels[i] = (j + 1);
        }
      }
    }
    return decisionTreeTrainLabels;
  }
}
