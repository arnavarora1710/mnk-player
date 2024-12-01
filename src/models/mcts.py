import numpy as np
import random
from enum import Enum

class Node:
    def __init__(self, state, parent=None):
        # state here is any game
        # but for our purposes it is an mnk game (consisting of m * n board)
        # with win condition being any row/column/diagonal having k consecutive blocks
        # of the same player
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    # fix zero division problem
    def best_child(self, exploration_weight=1.41):
        choices_weights = [
            (child.wins / child.visits if child.visits > 0 else float('inf')) +
            exploration_weight * np.sqrt(np.log(self.visits) / (child.visits + 1e-10))  
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
class SelectionStrategy(Enum):
    UCB1 = "UCB1"
    UCB1GRAVE = "UCB1GRAVE"
    ProgressiveHistory = "ProgressiveHistory"
    UCB1Tuned = "UCB1Tuned"

class ExplorationConst(Enum):
    CONST_0_1 = 0.1
    CONST_0_6 = 0.6
    CONST_1_41 = 1.41421356237

class PlayoutStrategy(Enum):
    Random200 = "Random200"
    MAST = "MAST"
    NST = "NST"

class ScoreBounds(Enum):
    TRUE = "true"
    FALSE = "false"


class MCTS:
    def __init__(self, game, strategy = "MCTS-UCB1GRAVE-0.1-NST-true"):
        self.game = game
        self.strategy = strategy
    
    def decode_strategy(self):
        # All agent string descriptions in training and test data are in 
        # the following format: MCTS-<SELECTION>-<EXPLORATION_CONST>-<PLAYOUT>-<SCORE_BOUNDS>

        # SELECTION: Selection strategy (UCB1, GRAVE, etc.)
        # EXPLORATION_CONST: Exploration constant (a float value)
        # PLAYOUT: Playout strategy (Random, Neural, etc.)
        # SCORE_BOUNDS: Whether to use score bounds (true or false)

        # Example: MCTS-UCB1-1.41-Random-true

        strategy = self.strategy.split("-")
        selection = SelectionStrategy(strategy[1])
        exploration_const = ExplorationConst(float(strategy[2]))
        playout = PlayoutStrategy(strategy[3])
        score_bounds = ScoreBounds(strategy[4])
        return selection, exploration_const, playout, score_bounds
    

    def _backpropagate(self, node, reward):
        _, _, _, score_bounds = self.decode_strategy()
        while node is not None:
            node.visits += 1
            if score_bounds == ScoreBounds.TRUE:
                node.wins += reward
            else:
                node.wins += reward / node.visits
            node = node.parent

    def _select(self, node):
        selection, exploration_const, _, _ = self.decode_strategy()
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                if selection == SelectionStrategy.UCB1:
                    node = node.best_child(exploration_const.value)
                elif selection == SelectionStrategy.UCB1GRAVE:
                    node = node.best_child(exploration_const.value)
                elif selection == SelectionStrategy.ProgressiveHistory:
                    node = node.best_child(exploration_const.value)
                elif selection == SelectionStrategy.UCB1Tuned:
                    node = node.best_child(exploration_const.value)
        return node

    def _simulate(self, state):
        _, _, playout, _ = self.decode_strategy()
        if playout == PlayoutStrategy.Random200:
            for _ in range(200):
                if state.is_terminal():
                    break
                legal_moves = state.get_legal_moves()
                move = random.choice(legal_moves)
                state = state.make_move(move)
        elif playout == PlayoutStrategy.MAST:
            while not state.is_terminal():
                legal_moves = state.get_legal_moves()
                move = random.choice(legal_moves)
                state = state.make_move(move)
        elif playout == PlayoutStrategy.NST:
            while not state.is_terminal():
                legal_moves = state.get_legal_moves()
                move = random.choice(legal_moves)
                state = state.make_move(move)
        return state.get_reward()

    def _backpropagate(self, node, reward):
        _, _, _, score_bounds = self.decode_strategy()
        while node is not None:
            node.visits += 1
            if score_bounds == ScoreBounds.TRUE:
                node.wins += reward
            else:
                node.wins += reward / node.visits
            node = node.parent

    def search(self, state, iterations=1000):
        root = Node(state)

        for _ in range(iterations):
            node = self._select(root)
            if node is None:
                continue
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        return root.best_child(0).state.get_last_move()

    def _expand(self, node):
        legal_moves = node.state.get_legal_moves()
        for move in legal_moves:
            new_state = node.state.make_move(move)
            if not any(child.state == new_state for child in node.children):
                child_node = Node(new_state, parent=node)
                node.children.append(child_node)
        return random.choice(node.children)