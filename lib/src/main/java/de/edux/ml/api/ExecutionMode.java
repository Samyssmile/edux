package de.edux.ml.api;

/**
 * This mode determines how batches are processed during training and testing.
 *
 * <p>Regardless of the chosen execution mode, all matrix operations are executed in parallel, with
 * the ExecutionMode parallelism referring to the processing of batches. Currently supported:
 *
 * <ul>
 *   <li>{@link #SINGLE_THREAD} - Single-thread execution mode, where batches are processed
 *       sequentially in a single thread.
 * </ul>
 */
public enum ExecutionMode {
  /**
   * Single-thread execution mode. In this mode, all batches are processed sequentially in a single
   * thread.
   */
  SINGLE_THREAD(1), MULTI_THREAD(6);

  int threads = 1;

  ExecutionMode(int threads) {
    this.threads = threads;
  }

  public int getThreads() {
    return threads;
  }

  public static ExecutionMode fromString(String mode) {
    if (mode.equalsIgnoreCase("single_thread")) {
      return SINGLE_THREAD;
    }
    throw new IllegalArgumentException("Unknown execution mode: " + mode);
  }
}
