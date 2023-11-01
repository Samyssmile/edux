package de.edux.ml.decisiontree;

class Node {
  double[][] data;
  double value;
  int splitFeature;
  Node left;
  Node right;
  boolean isLeaf;

  Node(double[][] data) {
    this.data = data;
    isLeaf = false;
  }
}
