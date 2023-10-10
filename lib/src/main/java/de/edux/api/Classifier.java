package de.edux.api;

public interface Classifier {

    /**
     * Trains the model using the provided training inputs and targets.
     * @param features 2D array of double, where each inner array represents
     * @param labels 2D array of double, where each inner array represents
     * @return true if the model was successfully trained, false otherwise.
     */
    boolean train(double[][] features, double[][] labels);
    /**
     * Evaluates the model's performance against the provided test inputs and targets.
     *
     * This method takes a set of test inputs and their corresponding expected targets,
     * applies the model to predict the outputs for the inputs, and then compares
     * the predicted outputs to the expected targets to evaluate the performance
     * of the model. The nature and metric of the evaluation (e.g., accuracy, MSE, etc.)
     * are dependent on the specific implementation within the method.
     *
     * @param testInputs 2D array of double, where each inner array represents
     *                   a single set of input values to be evaluated by the model.
     * @param testTargets 2D array of double, where each inner array represents
     *                    the expected output or target for the corresponding set
     *                    of inputs in {@code testInputs}.
     * @return a double value representing the performance of the model when evaluated
     *         against the provided test inputs and targets. The interpretation of this
     *         value (e.g., higher is better, lower is better, etc.) depends on the
     *         specific evaluation metric being used.
     * @throws IllegalArgumentException if the lengths of {@code testInputs} and
     *                                  {@code testTargets} do not match, or if
     *                                  they are empty.
     */
    double evaluate(double[][] testInputs, double[][] testTargets);


    /**
     * Predicts the output for a single set of input values.
     *
     * @param feature a single set of input values to be evaluated by the model.
     * @return a double array representing the predicted output values for the
     *         provided input values.
     * @throws IllegalArgumentException if {@code feature} is empty.
     */
    public double[] predict(double[] feature);

}
