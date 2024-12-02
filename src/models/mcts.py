import numpy as np
import random
from enum import Enum

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.41, grave_adjustment=False, tuned=False):
        if self.visits == 0:
            return random.choice(self.children)
        
        choices_weights = []
        for child in self.children:
            exploitation = child.wins / (child.visits + 1e-10)
            exploration = exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-10))
            
            if tuned:
                variance = np.var([c.wins / c.visits if c.visits > 0 else 0 for c in self.children])
                exploration *= np.sqrt(variance)
            
            if grave_adjustment:
                exploitation += self.compute_grave_adjustment(child)
            
            choices_weights.append(exploitation + exploration)

        return self.children[np.argmax(choices_weights)]

    def compute_grave_adjustment(self, child):
        # Example: Adjust using GRAVE heuristic (replace with domain-specific logic if needed)
        return 0.1 * child.wins / (child.visits + 1e-10)

class SelectionStrategy(Enum):
    UCB1 = "UCB1"
    UCB1GRAVE = "UCB1GRAVE"
    ProgressiveHistory = "ProgressiveHistory"
    UCB1Tuned = "UCB1Tuned"
    ProgressiveWidening = "ProgressiveWidening"

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
    def __init__(self, game, strategy="MCTS-UCB1GRAVE-0.1-NST-true"):
        self.game = game
        self.strategy = strategy

    def decode_strategy(self):
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
                    node = node.best_child(exploration_const.value, grave_adjustment=True)
                elif selection == SelectionStrategy.ProgressiveHistory:
                    node = node.best_child(exploration_const.value)
                elif selection == SelectionStrategy.UCB1Tuned:
                    node = node.best_child(exploration_const.value, tuned=True)
                elif selection == SelectionStrategy.ProgressiveWidening:
                    self._apply_progressive_widening(node)
                    node = node.best_child(exploration_const.value)
        return node

    def _apply_progressive_widening(self, node):
        if node.visits > len(node.children):
            self._expand(node)

    def _simulate(self, state):
        _, _, playout, _ = self.decode_strategy()
        steps = 200 if playout == PlayoutStrategy.Random200 else 100

        while not state.is_terminal() and steps > 0:
            steps -= 1
            legal_moves = state.get_legal_moves()
            move = random.choice(legal_moves)
            state = state.make_move(move)

        return state.get_reward()

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
