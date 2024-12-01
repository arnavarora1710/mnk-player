import numpy as np
import random

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


class MCTS:
    def __init__(self, game):
        self.game = game

    def search(self, state, iterations=1000):
        root = Node(state)

        for _ in range(iterations):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        return root.best_child(0).state.get_last_move()

    def _select(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                node = node.best_child()
        return node

    def _expand(self, node):
        legal_moves = node.state.get_legal_moves()
        for move in legal_moves:
            new_state = node.state.make_move(move)
            child_node = Node(new_state, parent=node)
            node.children.append(child_node)
        return random.choice(node.children)

    def _simulate(self, state):
        while not state.is_terminal():
            legal_moves = state.get_legal_moves()
            move = random.choice(legal_moves)
            state = state.make_move(move)
        return state.get_reward()

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent