package de.example.minimax;

import java.util.Scanner;

public class MinimaxOnTicTacToe {
    static boolean win = false;
    static String array[][] = new String[3][3];

    public static void main(String[] args) {
        int i = 0;
        while (!logic(array)) {
            if (i % 2 == 0) {
                System.out.println("Enter your position X : ");
                Scanner scanner = new Scanner(System.in);
                int position = Integer.parseInt(scanner.nextLine());
                input(position,"X");
            }
            else {
                System.out.println("O has made a move");

            }
            i ++;
            printLayout(array);
        }

    }


    private static void input(int position,String player) {
        switch (position){
            case 1 : array[0][0] = player;
                break;
            case 2 : array[0][1] = player;
                break;
            case 3 : array[0][2] = player;
                break;
            case 4 : array[1][0] = player;
                break;
            case 5 : array[1][1] = player;
                break;
            case 6 : array[1][2] = player;
                break;
            case 7 : array[2][0] = player;
                break;
            case 8 : array[2][1] = player;
                break;
            case 9 : array[2][2] = player;
                break;
        }
    }

    private static void printLayout(String[][] array) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                System.out.print(array[i][j] + "|");
            }
            System.out.print(array[i][2]);
            System.out.println();
        }
    }

    private static boolean logic(String[][] array) {
        if (array[0][0] == array[0][1] && array[0][1] == array[0][2] && array[0][1] != null){
            System.out.println("You won !!");
            return true;
        }

        if (array[1][0] == array[1][1] && array[1][1] == array[1][2] && array[1][1] != null){
            System.out.println("You won !!");
            return true;
        }
        if (array[2][0] == array[2][1] && array[2][1] == array[2][2] && array[2][1] != null){
            System.out.println("You won !!");
            return true;
        }
        if (array[0][0] == array[1][0] && array[1][0] == array[2][0] && array[1][0] != null){
            System.out.println("You won !!");
            return true;
        }

        if (array[0][1] == array[1][1] && array[1][1] == array[2][1] && array[1][1] != null){
            System.out.println("You won !!");
            return true;
        }

        if (array[0][2] == array[1][2] && array[1][2] == array[2][2] && array[1][2] != null){
            System.out.println("You won !!");
            return true;
        }

        if (array[0][0] == array[1][1] && array[1][1] == array[2][2] && array[1][1] != null){
            System.out.println("You won !!");
            return true;
        }
        if (array[0][2] == array[1][1] && array[1][1] == array[2][0] && array[1][1] != null){
            System.out.println("You won !!");
            return true;
        }
        return false;
    }


}
