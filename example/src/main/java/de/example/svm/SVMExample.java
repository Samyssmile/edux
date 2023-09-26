package de.example.svm;

import de.edux.ml.svm.ISupportVectorMachine;
import de.edux.ml.svm.SVMKernel;
import de.edux.ml.svm.SupportVectorMachine;
import de.example.data.iris.IrisProvider;

public class SVMExample {
    private static final boolean SHUFFLE = true;
    private static final boolean NORMALIZE = true;

    public static void main(String[] args){
        var datasetProvider = new IrisProvider(NORMALIZE, SHUFFLE, 0.6);
        datasetProvider.printStatistics();

        //Get Features and Labels
        double[][] features = datasetProvider.getTrainFeatures();
        // 1 - SATOSA 2 - VERSICOLOR 3 - VIRGINICA
        int[] labels = convert2DLabelArrayTo1DLabelArray(datasetProvider.getTrainLabels());


        ISupportVectorMachine supportVectorMachine = new SupportVectorMachine(SVMKernel.LINEAR, 1);
        //ONEvsONE Strategy
        supportVectorMachine.train(features, labels);

        double[][] testFeatures = datasetProvider.getTestFeatures();
        double[][] testLabels = datasetProvider.getTestLabels();
        int[] decisionTreeTestLabels = convert2DLabelArrayTo1DLabelArray(testLabels);

        supportVectorMachine.evaluate(testFeatures, decisionTreeTestLabels);
    }

    private static int[] convert2DLabelArrayTo1DLabelArray(double[][] labels) {
        int[] decisionTreeTrainLabels = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[i].length; j++) {
                if (labels[i][j] == 1) {
                    decisionTreeTrainLabels[i] = (j+1);
                }
            }
        }
        return decisionTreeTrainLabels;
    }
}
