package de.edux.ml.minmax;

import java.util.List;

public interface GameState<T> {
    List<T> getAvailableMoves();
    void applyMove(T move, T player);

    void applyMove(Integer move, Object player);

    void undoMove(T move);
    boolean isWinningState(T player);
    boolean isDraw();
}
