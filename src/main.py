from mnk import MNKGame
from models.mcts import MCTS

class MNKGameManager:
    def __init__(self, m, n, k, iterations=1000):
        self.game = MNKGame(m, n, k)
        self.mcts_agent = MCTS(self.game)
        self.iterations = iterations

    def display_instructions(self):
        print("Welcome to MNK Game!")
        print(f"Board size: {self.game.m}x{self.game.n}, Win condition: {self.game.k} consecutive marks.")
        print("You are Player 1 (X). MCTS Agent is Player 2 (O).")
        print("Enter your move as 'row col', where row and col are 0-indexed.")

    def get_human_move(self):
        move = None
        while move is None:
            try:
                user_input = input("Your move (row col): ")
                row, col = map(int, user_input.split())
                move = (row, col)
                self.game.make_move(move)
            except (ValueError, IndexError):
                print("Invalid input or move! Please try again.")
                move = None

    def play(self):
        self.display_instructions()
        while not self.game.is_terminal():
            self.game.display_board()

            if self.game.current_player == 1:
                print("Your Turn!")
                self.get_human_move()
            else:
                print("MCTS Agent is thinking...")
                mcts_move = self.mcts_agent.search(self.game, iterations=self.iterations)
                print(f"MCTS Agent chooses: {mcts_move}")
                self.game.make_move(mcts_move)

        self.game.display_board()
        self.display_results()

    def display_results(self):
        if self.game.is_terminal():
            print("Game Over!")
            if self.game.current_player == 1:
                print("MCTS Agent Wins!")
            else:
                print("Congratulations! You Win!")
        else:
            print("It's a Draw!")

if __name__ == "__main__":
    print("Set up your MNK game. Press Enter to use defaults (3x3 board, win with 3 marks).")
    try:
        m = int(input("Enter the number of rows (default: 3): ") or 3)
        n = int(input("Enter the number of columns (default: 3): ") or 3)
        k = int(input("Enter the win condition (default: 3): ") or 3)
        iterations = int(input("Enter MCTS iterations (default: 1000): ") or 1000)
    except ValueError:
        print("Invalid input. Using default values.")
        m, n, k, iterations = 3, 3, 3, 1000

    game_manager = MNKGameManager(m, n, k, iterations)
    game_manager.play()
