package de.edux.ml.minmax;

import java.util.function.Function;


public class MinMax<T> {

    private final GameState<T> gameState;
    private final T maximizingPlayer;
    private final T minimizingPlayer;
    private final int maxDepth;
    private final Function<GameState<T>, Integer> heuristic;

    /**
     * Constructs a MinMax object.
     *
     * @param gameState the current game state
     * @param maximizingPlayer the player aiming to maximize the evaluation score
     * @param minimizingPlayer the player aiming to minimize the evaluation score
     * @param maxDepth the maximum depth for the minimax search
     * @param heuristic the heuristic function to evaluate game states
     */
    public MinMax(GameState<T> gameState, T maximizingPlayer, T minimizingPlayer, int maxDepth, Function<GameState<T>, Integer> heuristic) {
        this.gameState = gameState;
        this.maximizingPlayer = maximizingPlayer;
        this.minimizingPlayer = minimizingPlayer;
        this.maxDepth = maxDepth;
        this.heuristic = heuristic;
    }



    /**
     * Determines the best move for the maximizing player using the minimax algorithm.
     *
     * @return the best possible move for the maximizing player
     */
    public T getBestMove() {
        int bestScore = Integer.MIN_VALUE;
        T bestMove = null;

        for (T move : gameState.getAvailableMoves()) {
            gameState.applyMove(move, maximizingPlayer);
            int score = minimax(gameState, 0, false);
            gameState.undoMove(move);

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        return bestMove;
    }

    /**
     * Recursive minimax algorithm implementation to evaluate game states.
     *
     * @param state the current game state
     * @param depth the current depth in the recursive search
     * @param isMaximizing true if the current player is the maximizing player, false otherwise
     * @return the evaluation score of the given game state
     */
    private int minimax(GameState<T> state, int depth, boolean isMaximizing) {
        if (state.isWinningState(maximizingPlayer)) return 1;
        if (state.isWinningState(minimizingPlayer)) return -1;
        if (state.isDraw()) return 0;

        if (depth == maxDepth) {
            return heuristic.apply(state);
        }

        if (isMaximizing) {
            int bestScore = Integer.MIN_VALUE;
            for (T move : state.getAvailableMoves()) {
                state.applyMove(move, maximizingPlayer);
                int score = minimax(state, depth + 1, false);
                state.undoMove(move);
                bestScore = Math.max(score, bestScore);
            }
            return bestScore;
        } else {
            int bestScore = Integer.MAX_VALUE;
            for (T move : state.getAvailableMoves()) {
                state.applyMove(move, minimizingPlayer);
                int score = minimax(state, depth + 1, true);
                state.undoMove(move);
                bestScore = Math.min(score, bestScore);
            }
            return bestScore;
        }
    }
}

