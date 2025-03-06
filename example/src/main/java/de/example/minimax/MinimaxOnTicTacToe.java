package de.example.minimax;

import java.util.Scanner;

// Min max algorithm on tic tac toe
public class MinimaxOnTicTacToe {

    private static String array[][] = new String[3][3];
    static boolean playing = true;
    static int a = 0;

    public static void main(String[] args) {
        while (playing) {
            if (a % 2 == 0) {
                System.out.println("Enter your position X : ");
                Scanner scanner = new Scanner(System.in);
                int position = Integer.parseInt(scanner.nextLine());
                if (isValidMove(position)) {
                    input(position, "X");
                    printLayout();
                    if (checkWinner(array, "X")) {
                        playing = false;
                        System.out.println("X is the winner");
                    }
                } else {
                    System.out.println("Invalid move. Position already taken or out of bounds.");
                    continue; // Skip incrementing a, so player gets another chance
                }
            }
            else {
                System.out.println();
                System.out.println("O makes a move");
                bestMove();
                printLayout();
                if (checkWinner(array,"O")){
                    playing = false;
                }
            }
            if (tie(array)){
                playing = false;
            }

            a ++;
        }
    }
    public static void bestMove() {
        int bestScore = Integer.MIN_VALUE;
        int[] move = new int[2];

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (array[i][j] == null) {
                    array[i][j] = "O";

                    int score = minimax(array,0, false);
                    array[i][j] = null;
                    if (score > bestScore) {
                        bestScore = score;
                        move[0] = i;
                        move[1] = j;
                    }
                }
            }
        }
        System.out.println(move[0]);
        System.out.println(move[1]);
        array[move[0]][move[1]] = "O";
    }

    private static boolean isValidMove(int position) {
        if (position < 1 || position > 9) {
            return false;
        }

        int row = (position - 1) / 3;
        int col = (position - 1) % 3;

        return array[row][col] == null;
    }

    private static void printLayout() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                System.out.print(array[i][j] + " | ");
            }
            System.out.print(array[i][2]);
            System.out.println();
        }
    }

    public static void input(int position, String player){
        int row = (position - 1) / 3;
        int col = (position - 1) % 3;
        array[row][col] = player;


    }

    public static boolean checkWinner(String array[][], String player){
        // Check rows
        for (int i = 0; i < 3; i++) {
            if (array[i][0] == player && array[i][1] == player && array[i][2] == player) {
                return true;
            }
        }

        // Check columns
        for (int i = 0; i < 3; i++) {
            if (array[0][i] == player && array[1][i] == player && array[2][i] == player) {
                return true;
            }
        }

        // Check diagonals
        if (array[0][0] == player && array[1][1] == player && array[2][2] == player) {
            return true;
        }
        if (array[0][2] == player && array[1][1] == player && array[2][0] == player) {
            return true;
        }

        return false;
    }

    public static boolean tie(String[][] array){
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (array[i][j] == null){
                    return false;
                }
            }
        }
        return true;
    }

    public static int minimax(String[][] array, int depth, boolean isMax){
//        System.out.println("--------------------------------------");
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 3; j++) {
//                System.out.print(array[i][j] + " ");
//            }
//            System.out.println();
//        }
        //Checking to see a winner
        if (checkWinner(array,"X")){
//            System.out.println("X won");
            return -1;
        }
        if (checkWinner(array,"O")){
//            System.out.println("O won");
            return 1;
        }
        if (tie(array)) return 0;

        // Ai player
        if (isMax) {
            int bestScore = Integer.MIN_VALUE;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (array[i][j] == null) {
                        array[i][j] = "O";
                        int score = minimax(array, depth + 1, false);
                        array[i][j] = null;
                        bestScore = Math.max(score,bestScore);
                    }
                }
            }
            return bestScore;
        }
        // Human player
        else {
            int bestScore = Integer.MAX_VALUE;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (array[i][j] == null) {
                        array[i][j] = "X";
                        int score = minimax(array, depth + 1, true);
                        array[i][j] = null;
                        bestScore = Math.min(score,bestScore);
                    }
                }
            }
            return bestScore;
        }
    }


}
