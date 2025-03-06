package de.example.minmax.chess;

// Datei: ChessGame.java

import de.edux.ml.minmax.GameState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;

public class ChessGame implements GameState<String> {

	private char[][]          board;
	private String            currentPlayer;
	private Stack<MoveRecord> moveHistory;

	public ChessGame() {
		board = new char[8][8];
		moveHistory = new Stack<>();
		initializeBoard();
		currentPlayer = "white";
	}

	// Initialisiert das Brett mit der Standardaufstellung
	private void initializeBoard() {
		// Zeile 0: Schwarze Figuren
		board[0] = new char[]{'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'};
		board[1] = new char[]{'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'};
		// Zeilen 2 bis 5: Leere Felder
		for (int i = 2; i <= 5; i++) {
			Arrays.fill(board[i], '.');
		}
		// Zeile 6: Weiße Bauern
		board[6] = new char[]{'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'};
		// Zeile 7: Weiße Figuren
		board[7] = new char[]{'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'};
	}

	// Prüft, ob (row, col) innerhalb des Brettes liegt.
	private boolean isInsideBoard(int row, int col) {
		return row >= 0 && row < 8 && col >= 0 && col < 8;
	}

	// Liefert true zurück, wenn die Figur (nicht leer) weiß ist.
	private boolean isWhite(char piece) {
		return Character.isUpperCase(piece);
	}

	// Gibt den Gegner zurück.
	private String getOpponent(String player) {
		return player.equals("white") ? "black" : "white";
	}

	// Bewertungsfunktion (Heuristik) anhand der Materialbilanz.
	public int evaluateBoard() {
		int score = 0;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				char piece = board[i][j];
				score += getPieceValue(piece);
			}
		}
		return score;
	}

	// Wertet eine Figur – Großbuchstaben = weiß, Kleinbuchstaben = schwarz.
	private int getPieceValue(char piece) {
		switch (Character.toLowerCase(piece)) {
			case 'p':
				return (piece == 'P') ? 1 : -1;
			case 'n':
				return (piece == 'N') ? 3 : -3;
			case 'b':
				return (piece == 'B') ? 3 : -3;
			case 'r':
				return (piece == 'R') ? 5 : -5;
			case 'q':
				return (piece == 'Q') ? 9 : -9;
			case 'k':
				return (piece == 'K') ? 100 : -100; // König bekommt einen hohen Wert
			default:
				return 0;
		}
	}

	// Implementierung von getAvailableMoves aus GameState.
	// Hier werden nur die Züge der Figuren generiert, die aktuell am Zug sind.
	// Die Züge werden in algebraischer Notation (z.B. "e2e4") kodiert.
	@Override
	public List<String> getAvailableMoves() {
		List<String> moves = new ArrayList<>();
		// Durchlaufe das Brett und suche nach Figuren des aktiven Spielers.
		for (int row = 0; row < 8; row++) {
			for (int col = 0; col < 8; col++) {
				char piece = board[row][col];
				if (piece == '.') continue;
				if (currentPlayer.equals("white") && !isWhite(piece)) continue;
				if (currentPlayer.equals("black") && isWhite(piece)) continue;
				// Zuggenerierung je nach Figur
				switch (Character.toLowerCase(piece)) {
					case 'p':
						moves.addAll(generatePawnMoves(row, col, piece));
						break;
					case 'n':
						moves.addAll(generateKnightMoves(row, col, piece));
						break;
					case 'k':
						moves.addAll(generateKingMoves(row, col, piece));
						break;
					case 'r':
						moves.addAll(generateSlidingMoves(row,
						                                  col,
						                                  piece,
						                                  new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}));
						break;
					case 'b':
						moves.addAll(generateSlidingMoves(row,
						                                  col,
						                                  piece,
						                                  new int[][]{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}));
						break;
					case 'q':
						moves.addAll(generateSlidingMoves(row, col, piece, new int[][]{
								{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
						}));
						break;
					// Weitere Figuren können hier ergänzt werden
				}
			}
		}
		return moves;
	}

	// Generiert Züge für Bauern.
	private List<String> generatePawnMoves(int row, int col, char pawn) {
		List<String> moves     = new ArrayList<>();
		int          direction = isWhite(pawn) ? -1 : 1; // Weiße Bauern bewegen sich "nach oben", schwarze "nach unten"
		int          nextRow   = row + direction;
		// Vorwärtszug, wenn das Zielfeld leer ist.
		if (isInsideBoard(nextRow, col) && board[nextRow][col] == '.') {
			moves.add(encodeMove(row, col, nextRow, col));
		}
		// Diagonale Schläge
		for (int dcol = -1; dcol <= 1; dcol += 2) {
			int newCol = col + dcol;
			if (isInsideBoard(nextRow, newCol)) {
				char target = board[nextRow][newCol];
				if (target != '.' && isWhite(target) != isWhite(pawn)) {
					moves.add(encodeMove(row, col, nextRow, newCol));
				}
			}
		}
		return moves;
	}

	// Generiert Züge für Springer.
	private List<String> generateKnightMoves(int row, int col, char knight) {
		List<String> moves   = new ArrayList<>();
		int[][]      offsets = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
		for (int[] off : offsets) {
			int newRow = row + off[0], newCol = col + off[1];
			if (isInsideBoard(newRow, newCol)) {
				char target = board[newRow][newCol];
				if (target == '.' || isWhite(target) != isWhite(knight)) {
					moves.add(encodeMove(row, col, newRow, newCol));
				}
			}
		}
		return moves;
	}

	// Generiert Züge für den König.
	private List<String> generateKingMoves(int row, int col, char king) {
		List<String> moves   = new ArrayList<>();
		int[][]      offsets = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
		for (int[] off : offsets) {
			int newRow = row + off[0], newCol = col + off[1];
			if (isInsideBoard(newRow, newCol)) {
				char target = board[newRow][newCol];
				if (target == '.' || isWhite(target) != isWhite(king)) {
					moves.add(encodeMove(row, col, newRow, newCol));
				}
			}
		}
		return moves;
	}

	// Generiert Züge für Schieber (Turm, Läufer, Dame) anhand gegebener Richtungen.
	private List<String> generateSlidingMoves(int row, int col, char piece, int[][] directions) {
		List<String> moves = new ArrayList<>();
		for (int[] dir : directions) {
			int newRow = row, newCol = col;
			while (true) {
				newRow += dir[0];
				newCol += dir[1];
				if (!isInsideBoard(newRow, newCol))
					break;
				char target = board[newRow][newCol];
				if (target == '.') {
					moves.add(encodeMove(row, col, newRow, newCol));
				} else {
					if (isWhite(target) != isWhite(piece)) {
						moves.add(encodeMove(row, col, newRow, newCol));
					}
					break;
				}
			}
		}
		return moves;
	}

	// Kodiert einen Zug in algebraischer Notation (z. B. "e2e4").
	private String encodeMove(int fromRow, int fromCol, int toRow, int toCol) {
		return "" + (char) ('a' + fromCol) + (8 - fromRow)
				+ (char) ('a' + toCol) + (8 - toRow);
	}

	// Parst einen Zugstring (z. B. "e2e4") in Koordinaten.
	private int[] parseMove(String move) {
		int fromCol = move.charAt(0) - 'a';
		int fromRow = 8 - (move.charAt(1) - '0');
		int toCol   = move.charAt(2) - 'a';
		int toRow   = 8 - (move.charAt(3) - '0');
		return new int[]{fromRow, fromCol, toRow, toCol};
	}

	// Wendet einen Zug an. Der Parameter 'player' wird ignoriert, da intern der currentPlayer genutzt wird.
	@Override
	public void applyMove(String move, String player) {
		int[] coords        = parseMove(move);
		int   fromRow       = coords[0], fromCol = coords[1], toRow = coords[2], toCol = coords[3];
		char  movedPiece    = board[fromRow][fromCol];
		char  capturedPiece = board[toRow][toCol];
		// Aktualisiere das Brett
		board[toRow][toCol] = movedPiece;
		board[fromRow][fromCol] = '.';
		// Speichere den Zug, um ihn später rückgängig machen zu können
		moveHistory.push(new MoveRecord(fromRow, fromCol, toRow, toCol, movedPiece, capturedPiece, currentPlayer));
		// Wechsle den aktiven Spieler
		currentPlayer = getOpponent(currentPlayer);
	}

	@Override
	public void applyMove(Integer move, String player) {

	}

	// Macht einen Zug rückgängig.
	@Override
	public void undoMove(String move) {
		if (moveHistory.isEmpty()) return;
		MoveRecord record = moveHistory.pop();
		board[record.fromRow][record.fromCol] = record.movedPiece;
		board[record.toRow][record.toCol] = record.capturedPiece;
		currentPlayer = record.previousPlayer;
	}

	// Prüft, ob für einen Spieler der Gewinnzustand erreicht ist,
	// d. h. ob der gegnerische König fehlt.
	@Override
	public boolean isWinningState(String player) {
		// Wenn "white" am Zug ist, prüfe, ob der schwarze König (k) fehlt, und umgekehrt.
		char kingToFind = player.equals("white") ? 'k' : 'K';
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				if (board[i][j] == kingToFind)
					return false;
			}
		}
		return true;
	}

	// Für diese vereinfachte Version wird Remis nicht weiter betrachtet.
	@Override
	public boolean isDraw() {
		return false;
	}

	// Gibt das Brett in der Konsole aus (zur Debug-Ausgabe).
	public void printBoard() {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				System.out.print(board[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println("Aktiver Spieler: " + currentPlayer);
	}

	// Innere Klasse zur Speicherung von Zuginformationen für das Undo.
	private class MoveRecord {
		int fromRow, fromCol, toRow, toCol;
		char movedPiece, capturedPiece;
		String previousPlayer;

		MoveRecord(int fromRow, int fromCol, int toRow, int toCol,
		           char movedPiece, char capturedPiece, String previousPlayer) {
			this.fromRow = fromRow;
			this.fromCol = fromCol;
			this.toRow = toRow;
			this.toCol = toCol;
			this.movedPiece = movedPiece;
			this.capturedPiece = capturedPiece;
			this.previousPlayer = previousPlayer;
		}
	}
}
