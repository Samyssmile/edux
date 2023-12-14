package de.edux.ml.mlp.exceptions;

/**
 * This class represents an exception that is thrown when an unsupported
 * layer type is encountered in the MLP context. It extends RuntimeException
 * to indicate that this is an unchecked exception that might occur during
 * the runtime of the application, particularly when configuring or building
 * MLP models with incompatible or unsupported layer types.
 */
public class UnsupportedLayerException extends RuntimeException {

    /**
     * Constructs a new UnsupportedLayerException with the default message.
     */
    public UnsupportedLayerException() {
        super("The specified layer type is not supported.");
    }

    /**
     * Constructs a new UnsupportedLayerException with a custom message.
     *
     * @param message the detail message. The detail message is saved for
     *                later retrieval by the Throwable.getMessage() method.
     */
    public UnsupportedLayerException(String message) {
        super(message);
    }
}
