package de.example.minmax.tictactoe;

import de.edux.ml.minmax.GameState;

import java.util.ArrayList;
import java.util.List;

public class TicTacToe implements GameState<Integer> {

	private final String[][] board = new String[3][3];

	@Override
	public List<Integer> getAvailableMoves() {
		List<Integer> moves = new ArrayList<>();
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (board[i][j] == null) {
					moves.add(i * 3 + j + 1);
				}
			}
		}
		return moves;
	}

	@Override
	public void applyMove(Integer position, Integer player) {
		int row = (position - 1) / 3;
		int col = (position - 1) % 3;
		board[row][col] = player == 1 ? "X" : "O";
	}

	@Override
	public void undoMove(Integer position) {
		int row = (position - 1) / 3;
		int col = (position - 1) % 3;
		board[row][col] = null;
	}

	@Override
	public boolean isWinningState(Integer player) {
		String symbol = player == 1 ? "X" : "O";
		return checkWinner(symbol);
	}

	@Override
	public boolean isDraw() {
		return getAvailableMoves().isEmpty();
	}

	private boolean checkWinner(String player) {
		for (int i = 0; i < 3; i++) {
			if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return true;
			if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return true;
		}
		return (board[0][0] == player && board[1][1] == player && board[2][2] == player) ||
				(board[0][2] == player && board[1][1] == player && board[2][0] == player);
	}

	public void printBoard() {
		System.out.println("-------------");
		for (int i = 0; i < 3; i++) {
			System.out.print("| ");
			for (int j = 0; j < 3; j++) {
				System.out.print((board[i][j] == null ? " " : board[i][j]) + " | ");
			}
			System.out.println("\n-------------");
		}
	}

}
