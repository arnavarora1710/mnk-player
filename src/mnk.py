class MNKGame:
    def __init__(self, m, n, k):
        self.m = m  # Number of rows
        self.n = n  # Number of columns
        self.k = k  # Number of consecutive marks needed to win
        self.board = [[[None for _ in range(n)] for _ in range(m)] for _ in range(k)]
        self.current_player = 1  # Player 1 starts

    def get_legal_moves(self):
        legal_moves = []
        for i in range(self.m):
            for j in range(self.n):
                if self.board[i][j] is None:  # Check if the cell is empty
                    legal_moves.append((i, j))
        return legal_moves

    def make_move(self, move):
        row, col = move
        if self.board[row][col] is not None:
            raise ValueError("Invalid move: Cell is already occupied.")
        self.board[row][col] = self.current_player
        self.current_player = 2 if self.current_player == 1 else 1  # Switch player

    def is_terminal(self):
        # Check for a win condition
        for i in range(self.m):
            for j in range(self.n):
                if self.board[i][j] is not None:
                    if self.check_win(i, j):
                        return True
        return False

    def check_win(self, row, col):
        player = self.board[row][col]
        # Check all directions for a win
        return (self.check_direction(row, col, 1, 0, player) or  # Horizontal
                self.check_direction(row, col, 0, 1, player) or  # Vertical
                self.check_direction(row, col, 1, 1, player) or  # Diagonal \
                self.check_direction(row, col, 1, -1, player))   # Diagonal /

    def check_direction(self, row, col, delta_row, delta_col, player):
        count = 0
        for d in range(-self.k + 1, self.k):  # Check in both directions
            r = row + d * delta_row
            c = col + d * delta_col
            if 0 <= r < self.m and 0 <= c < self.n and self.board[r][c] == player:
                count += 1
                if count == self.k:
                    return True
            else:
                count = 0
        return False
    
    def reset_game(self):
        self.board = [[None for _ in range(self.n)] for _ in range(self.m)]
        self.current_player = 1

    def save_state(self, filename):
        import json
        state = {
            "board": self.board,
            "current_player": self.current_player,
            "m": self.m,
            "n": self.n,
            "k": self.k
        }
        with open(filename, "w") as f:
            json.dump(state, f)
        print(f"Game state saved to {filename}.")

    def load_state(self, filename):
        import json
        with open(filename, "r") as f:
            state = json.load(f)
        self.board = state["board"]
        self.current_player = state["current_player"]
        self.m = state["m"]
        self.n = state["n"]
        self.k = state["k"]
        print(f"Game state loaded from {filename}.")

    def display_board(self):
        for row in self.board:
            print(" ".join([str(cell) if cell is not None else "." for cell in row]))
        print()
