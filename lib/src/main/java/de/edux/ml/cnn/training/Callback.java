package de.edux.ml.cnn.training;

public interface Callback {
    default void onEpochStart(TrainerContext ctx) {}
    default void onBatchEnd(TrainerContext ctx) {}
    default void onEpochEnd(TrainerContext ctx) {}
}