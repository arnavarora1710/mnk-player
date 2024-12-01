import numpy as np
import json

class MNKGame:
    def __init__(self, m, n, k):
        self.m = m  # Number of rows
        self.n = n  # Number of columns
        self.k = k  # Number of consecutive marks needed to win
        self.board = np.full((m, n), ' ')  # Empty board
        self.current_player = 'X'  # Player X starts
        self.winner = None  # To store the winner
        self.last_move = None  # To store the last move

    def display_board(self):
        print("\n=== Current Board ===")
        print("   " + "   ".join(f"{i:2}" for i in range(self.n)))  # Column indices
        print("  +" + "---+" * self.n)
        for i, row in enumerate(self.board):
            row_display = " | ".join(row)
            print(f"{i:2} | {row_display} |")  # Row index and row content
            print("  +" + "---+" * self.n)
        print(f"Next Player: {self.current_player}\n")

    def get_legal_moves(self):
        return [(i, j) for i in range(self.m) for j in range(self.n) if self.board[i, j] == ' ']

    def make_move(self, move):
        row, col = move
        if self.board[row, col] != ' ':
            raise ValueError("Invalid move: Cell is already occupied.")
        new_game = self.copy()
        new_game.board[row, col] = self.current_player
        new_game.last_move = move
        new_game.current_player = 'O' if self.current_player == 'X' else 'X'
        new_game.check_winner(row, col)
        return new_game

    def is_terminal(self):
        """Check if the game is over."""
        if self.winner is not None:
            return True  # Someone has won
        return all(cell != ' ' for row in self.board for cell in row)  # Draw

    def get_reward(self):
        """Return the reward for the current state."""
        if self.winner == 'X':
            return 1
        elif self.winner == 'O':
            return -1
        return 0  # Draw or ongoing game

    def check_winner(self, row, col):
        """Check if the last move resulted in a win."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            if self.check_direction(row, col, dx, dy, player):
                self.winner = player
                return
        self.winner = None

    def check_direction(self, row, col, dx, dy, player):
        count = 0
        for step in range(-self.k + 1, self.k):
            r, c = row + step * dx, col + step * dy
            if 0 <= r < self.m and 0 <= c < self.n and self.board[r, c] == player:
                count += 1
                if count == self.k:
                    return True
            else:
                count = 0
        return False
    
    def get_last_move(self):
        return self.last_move

    def copy(self):
        new_game = MNKGame(self.m, self.n, self.k)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        new_game.winner = self.winner
        return new_game

    def save_state(self, filename):
        state = {
            "board": self.board.tolist(),
            "current_player": self.current_player,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "winner": self.winner,
            "last_move": self.last_move
        }
        with open(filename, "w") as f:
            json.dump(state, f)
        print(f"Game state saved to {filename}.")

    def load_state(self, filename):
        with open(filename, "r") as f:
            state = json.load(f)
        self.board = np.array(state["board"])
        self.current_player = state["current_player"]
        self.m = state["m"]
        self.n = state["n"]
        self.k = state["k"]
        self.winner = state["winner"]
        self.last_move = tuple(state["last_move"])
        print(f"Game state loaded from {filename}.")

    def reset_game(self):
        self.board = np.full((self.m, self.n), ' ')
        self.current_player = 'X'
        self.winner = None
        self.last_move = None
