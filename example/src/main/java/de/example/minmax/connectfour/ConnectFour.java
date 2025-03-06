package de.example.minmax.connectfour;

import de.edux.ml.minmax.MinMax;

import java.util.Scanner;

public class ConnectFour {

	public static void main(String[] args) {
		Scanner         scanner       = new Scanner(System.in);
		ConnectFourGame game          = new ConnectFourGame();
		int             currentPlayer = 1;

		while (true) {
			System.out.println();
			game.printBoard();

			if (game.isWinningState(1)) {
				System.out.println("The AI wins!");
				break;
			} else if (game.isWinningState(2)) {
				System.out.println("You won!");
				break;
			} else if (game.isDraw()) {
				System.out.println("It's a draw!");
				break;
			}

			if (currentPlayer == 1) {
				System.out.println("AI is making a move...");
				MinMax<Integer> ai = new MinMax<>(
						game,
						1,
						2,
						6,
						state -> {
							ConnectFourGame cf = (ConnectFourGame) state;
							return cf.evaluateHeuristic(1, 2);
						}
				);
				Integer move = ai.getBestMove();
				System.out.println("AI selects column: " + move);
				game.applyMove(move, 1);
			} else {
				System.out.print("Your turn. Choose a column (0-6): ");
				int move = scanner.nextInt();
				if (!game.getAvailableMoves().contains(move)) {
					System.out.println("Invalid move. Please try again.");
					continue;
				}
				game.applyMove(move, 2);
			}

			currentPlayer = (currentPlayer == 1) ? 2 : 1;
		}
		System.out.println();
		game.printBoard();
		scanner.close();
	}
}