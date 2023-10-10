package de.example.svm;

import de.edux.api.Classifier;
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
        var features = datasetProvider.getTrainFeatures();
        // 1 - SATOSA 2 - VERSICOLOR 3 - VIRGINICA
        var labels = datasetProvider.getTrainLabels();


        Classifier supportVectorMachine = new SupportVectorMachine(SVMKernel.LINEAR, 1);
        //ONEvsONE Strategy
        supportVectorMachine.train(features, labels);

        double[][] testFeatures = datasetProvider.getTestFeatures();
        double[][] testLabels = datasetProvider.getTestLabels();

        supportVectorMachine.evaluate(testFeatures, testLabels);
    }


}
