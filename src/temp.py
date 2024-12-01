import numpy as np
import random
from math import sqrt, log
from models.mcts import MCTS
from mnk import MNKGame

def run_simulation():
    game = MNKGame(7, 7, 4)
    mcts = MCTS(game)

    print("=== Simulation Start ===")
    print("Player X: MCTS Agent")
    print("Player O: Random Agent")
    print()

    while not game.is_terminal():
        if game.current_player == 'X':
            move = mcts.search(game, iterations=500)  
            print("MCTS Agent (X) chooses:", move)
        else:
            move = random.choice(game.get_legal_moves())  
            print("Random Agent (O) chooses:", move)

        game = game.make_move(move)  
        game.display_board()

        if game.is_terminal():
            break

    if game.winner:
        print(f"Winner: {game.winner} ({'MCTS Agent' if game.winner == 'X' else 'Random Agent'})")
    else:
        print("Result: Draw")

if __name__ == "__main__":
    run_simulation()
