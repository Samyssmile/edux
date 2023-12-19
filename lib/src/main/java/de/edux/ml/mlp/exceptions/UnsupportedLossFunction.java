package de.edux.ml.mlp.exceptions;

/**
 * Represents an exception that is thrown when an unsupported loss function is encountered.
 * This class is part of the MLP (Multi-Layer Perceptron) framework and is used to signal
 * that the specified loss function is not supported by the current implementation.
 *
 * It extends RuntimeException, which allows this exception to be thrown and propagated
 * through the call stack without being explicitly declared in method signatures.
 */
public class UnsupportedLossFunction extends RuntimeException {

    /**
     * Constructs a new UnsupportedLossFunction exception with the default message.
     */
    public UnsupportedLossFunction() {
        super("Unsupported loss function.");
    }

    /**
     * Constructs a new UnsupportedLossFunction exception with a custom message.
     *
     * @param message The custom message that describes this exception.
     */
    public UnsupportedLossFunction(String message) {
        super(message);
    }
}
