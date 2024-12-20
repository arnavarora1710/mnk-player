import time
import numpy as np
import random
from math import sqrt, log
from models.mcts import MCTS
from mnk import MNKGame
from inference import get_best_string
from pipeline import generate_ludii_mnk_game

def test(m, n, k):
    total_time = 0
    mcts_moves_played = 0
    game = MNKGame(m, n, k, 'X')
    generate_ludii_mnk_game(m, n, k)
    # if m == 4:
    optimal_strat = get_best_string()
    # else:
    #     optimal_strat = "MCTS-UCB1-1.41421356237-Random200-true"
    mcts = MCTS(game, optimal_strat)

    print("=== Simulation Start ===")
    print("Player X: MCTS Agent")
    print("Player O: Our Agent")
    print()

    while not game.is_terminal():
        if game.current_player == 'X':
            # calculate total time taken for MCTS to make a move
            start_time = time.time()
            move = mcts.search(game, iterations=1000)  
            end_time = time.time()
            total_time += end_time - start_time
            print("Time taken for MCTS to make a move:", end_time - start_time)
            mcts_moves_played += 1
            print("MCTS Agent (X) chooses:", move)
        else:
            move = tuple(map(int, input("Enter move (row, col): ").split()))
            print("Our Agent (O) chooses:", move)

        game = game.make_move(move)  
        game.display_board()

        if game.is_terminal():
            break

    if game.winner:
        print(f"Winner: {game.winner} ({'MCTS Agent' if game.winner == 'X' else 'Our Agent'})")
    else:
        print("Result: Draw")
    print("Total time taken for MCTS to make all moves:", total_time)
    print("Average time taken for MCTS to make a move:", total_time / mcts_moves_played)
    print("=== Simulation End ===")

if __name__ == "__main__":
    # take m, n, k as input
    m, n, k = map(int, input("Enter m, n, k (space separated): ").split())
    print("Test Case [m: {}, n: {}, k: {}]".format(m, n, k))
    test(m, n, k)
