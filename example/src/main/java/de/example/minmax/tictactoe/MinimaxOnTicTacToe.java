package de.example.minmax.tictactoe;

import de.edux.ml.minmax.MinMax;

public class MinimaxOnTicTacToe {

	private final static int MAXIMIZING_PLAYER = 1;
	private final static int MINIMIZING_PLAYER = 2;
	private final static int DEPTH             = 3;

	public static void main(String[] args) {
		TicTacToe ticTacToe = getTicTacToePosition();
		System.out.println("Starting position:");
		ticTacToe.printBoard();

		MinMax<Integer> minMax = new MinMax<>(
				ticTacToe,
				MAXIMIZING_PLAYER,
				MINIMIZING_PLAYER,
				DEPTH,
				state -> {
					// For this simple example, we return a neutral value.
					return 0;
				}
		);

		int bestMove = minMax.getBestMove();
		System.out.println("Best move for AI: " + bestMove);
		//apply best move
		ticTacToe.applyMove(bestMove, MAXIMIZING_PLAYER);
		ticTacToe.printBoard();
	}

	public static TicTacToe getTicTacToePosition() {
		TicTacToe ticTacToe = new TicTacToe();
		ticTacToe.applyMove(3, MINIMIZING_PLAYER);
		ticTacToe.applyMove(1, MAXIMIZING_PLAYER);
		ticTacToe.applyMove(5, MINIMIZING_PLAYER);
		ticTacToe.applyMove(7, MAXIMIZING_PLAYER);
		ticTacToe.applyMove(4, MINIMIZING_PLAYER);

		return ticTacToe;
	}
}



