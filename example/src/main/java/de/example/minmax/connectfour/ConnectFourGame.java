package de.example.minmax.connectfour;

import de.edux.ml.minmax.GameState;

import java.util.ArrayList;
import java.util.List;

class ConnectFourGame implements GameState<Integer> {
	private final int     ROWS = 6;
	private final int     COLS = 7;
	private       int[][] board;

	public ConnectFourGame() {
		board = new int[ROWS][COLS];
	}

	@Override
	public List<Integer> getAvailableMoves() {
		List<Integer> moves = new ArrayList<>();
		for (int c = 0; c < COLS; c++) {
			if (board[0][c] == 0) {
				moves.add(c);
			}
		}
		return moves;
	}

	@Override
	public void applyMove(Integer move, Integer player) {
		int col = move;
		for (int r = ROWS - 1; r >= 0; r--) {
			if (board[r][col] == 0) {
				board[r][col] = player;
				break;
			}
		}
	}

	@Override
	public void undoMove(Integer move) {
		int col = move;
		for (int r = 0; r < ROWS; r++) {
			if (board[r][col] != 0) {
				board[r][col] = 0;
				break;
			}
		}
	}

	@Override
	public boolean isWinningState(Integer player) {
		for (int r = 0; r < ROWS; r++) {
			for (int c = 0; c <= COLS - 4; c++) {
				if (board[r][c] == player && board[r][c + 1] == player &&
						board[r][c + 2] == player && board[r][c + 3] == player) {
					return true;
				}
			}
		}
		for (int c = 0; c < COLS; c++) {
			for (int r = 0; r <= ROWS - 4; r++) {
				if (board[r][c] == player && board[r + 1][c] == player &&
						board[r + 2][c] == player && board[r + 3][c] == player) {
					return true;
				}
			}
		}
		for (int r = 0; r <= ROWS - 4; r++) {
			for (int c = 0; c <= COLS - 4; c++) {
				if (board[r][c] == player && board[r + 1][c + 1] == player &&
						board[r + 2][c + 2] == player && board[r + 3][c + 3] == player) {
					return true;
				}
			}
		}
		for (int r = 3; r < ROWS; r++) {
			for (int c = 0; c <= COLS - 4; c++) {
				if (board[r][c] == player && board[r - 1][c + 1] == player &&
						board[r - 2][c + 2] == player && board[r - 3][c + 3] == player) {
					return true;
				}
			}
		}
		return false;
	}

	@Override
	public boolean isDraw() {
		return getAvailableMoves().isEmpty();
	}

	public void printBoard() {
		for (int r = 0; r < ROWS; r++) {
			for (int c = 0; c < COLS; c++) {
				int  cell   = board[r][c];
				char symbol = (cell == 0) ? '.' : (cell == 1 ? 'X' : 'O');
				System.out.print(symbol + " ");
			}
			System.out.println();
		}
		for (int c = 0; c < COLS; c++) {
			System.out.print(c + " ");
		}
		System.out.println();
	}

	public int evaluateHeuristic(int maximizingPlayer, int minimizingPlayer) {
		int score        = 0;
		int centerColumn = COLS / 2;
		int centerCount  = 0;
		for (int r = 0; r < ROWS; r++) {
			if (board[r][centerColumn] == maximizingPlayer) {
				centerCount++;
			}
		}
		score += centerCount * 3;

		for (int r = 0; r < ROWS; r++) {
			for (int c = 0; c < COLS - 3; c++) {
				int[] window = new int[4];
				for (int i = 0; i < 4; i++) {
					window[i] = board[r][c + i];
				}
				score += evaluateWindow(window, maximizingPlayer, minimizingPlayer);
			}
		}

		return score;
	}

	private int evaluateWindow(int[] window, int maximizingPlayer, int minimizingPlayer) {
		int score = 0, countMax = 0, countMin = 0, countEmpty = 0;

		for (int cell : window) {
			if (cell == maximizingPlayer) countMax++;
			else if (cell == minimizingPlayer) countMin++;
			else countEmpty++;
		}

		if (countMax == 4) score += 100;
		else if (countMax == 3 && countEmpty == 1) score += 5;
		else if (countMax == 2 && countEmpty == 2) score += 2;

		if (countMin == 3 && countEmpty == 1) score -= 4;

		return score;
	}
}
