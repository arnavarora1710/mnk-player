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
    
    def print_state(self):
        self.state.display_board()

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.41, grave_adjustment=False, tuned=False):
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
    def __init__(self, game, strategy="MCTS-UCB1-0.1-Random200-true"):
        self.game = game
        self.strategy = strategy
        self.root = None

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
        if not node.is_fully_expanded():
            self._expand(node)
        if selection == SelectionStrategy.UCB1:
            return node.best_child(exploration_const.value)
        elif selection == SelectionStrategy.UCB1GRAVE:
            return node.best_child(exploration_const.value, grave_adjustment=True)
        elif selection == SelectionStrategy.ProgressiveHistory:
            return node.best_child(exploration_const.value)
        elif selection == SelectionStrategy.UCB1Tuned:
            return node.best_child(exploration_const.value, tuned=True)
        elif selection == SelectionStrategy.ProgressiveWidening:
            self._apply_progressive_widening(node)
            return node.best_child(exploration_const.value)
        else:
            return random.choice(node.children)

    def _apply_progressive_widening(self, node):
        if node.visits > len(node.children):
            self._expand(node)

    def _simulate(self, state):
        _, _, playout, _ = self.decode_strategy()
        steps = 200 if playout == PlayoutStrategy.Random200 else 100

        while not state.is_terminal() and steps > 0:
            steps -= 1
            legal_moves = state.get_legal_moves()
            center = (state.m // 2, state.n // 2)
            legal_moves.sort(key=lambda move: (move[0] - center[0]) ** 2 + (move[1] - center[1]) ** 2)
            # give more weight to earlier moves in random choice (make up probabilities that are higher towards the start) (towards center)
            probabilities = [1 / (i + 1) for i in range(len(legal_moves))]
            probabilities = [p / sum(probabilities) for p in probabilities]
            move = random.choices(legal_moves, weights=probabilities)[0]
            # move = random.choice(legal_moves)
            state = state.make_move(move)

        return state.get_reward()

    def search(self, state, iterations=1000):
        self.root = Node(state)

        for _ in range(iterations):
            node = self._select(self.root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        # Select the child with the highest number of wins
        best_child = max(self.root.children, key=lambda child: child.wins)
        return best_child.state.get_last_move()

    def _expand(self, node):
        legal_moves = node.state.get_legal_moves()
        for move in legal_moves:
            new_state = node.state.make_move(move)
            if not any(child.state == new_state for child in node.children):
                child_node = Node(new_state, parent=node)
                node.children.append(child_node)