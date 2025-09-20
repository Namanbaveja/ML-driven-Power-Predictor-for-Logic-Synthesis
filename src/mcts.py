# aig_qor/mcts.py

import math
import random
from typing import Optional, List

import torch

from .utils import predict_design_qor


class ModelMCTSNode:
    """
    Node used by ModelMCTS representing a partial recipe.
    """
    def __init__(self, transformation: Optional[str], depth: int, recipe: List[str], parent: Optional["ModelMCTSNode"] = None):
        self.transformation = transformation
        self.depth = depth
        self.recipe = recipe
        self.parent = parent
        self.children: List[ModelMCTSNode] = []
        self.visits: int = 0
        self.reward: float = 0.0
        self.power_value: Optional[float] = None

    def add_child(self, command: str) -> Optional["ModelMCTSNode"]:
        """
        Add a new child with transformation `command`. If the command already exists among children,
        returns None.
        """
        if command in [c.transformation for c in self.children]:
            return None
        new_recipe = self.recipe + [command]
        child = ModelMCTSNode(command, self.depth + 1, new_recipe, self)
        self.children.append(child)
        return child

    def get_recipe_str(self) -> str:
        """Return the recipe as a semicolon-separated string (ending with ';' if non-empty)."""
        return ";".join(self.recipe) + ";" if self.recipe else ""


class ModelMCTS:
    """
    Monte Carlo Tree Search that uses a learned model to evaluate candidate recipes.

    The model should be callable as:
        predict_design_qor(model, design_name, recipe_str, std, mean, vocab, ...)
    returning a list of predicted QoR values (one per step) where the last element is the final QoR.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        design: str,
        qor_stats: dict,
        vocab: dict,
        aigs_dir: str,
        max_recipe_len: int,
        first_power: float,
        mean: float,
        std: float,
        device: torch.device
    ):
        self.root = ModelMCTSNode(transformation=None, depth=0, recipe=[])
        self.model = model.to(device).eval()
        self.design = design
        self.qor_stats = qor_stats
        self.vocab = vocab
        self.aigs_dir = aigs_dir
        self.max_recipe_len = max_recipe_len
        self.first_power = first_power
        self.global_mean = mean
        self.global_std = std
        self.device = device
        self.best_node: Optional[ModelMCTSNode] = None

    def uct_score(self, node: ModelMCTSNode) -> float:
        """
        Upper Confidence bound for Trees (UCT) score.
        If node has zero visits or parent is None, return +inf to force exploration.
        """
        if node.parent is None or node.visits == 0:
            return float("inf")
        parent_visits = max(node.parent.visits, 1)
        exploitation = node.reward / node.visits
        exploration = math.sqrt(2 * math.log(parent_visits) / node.visits)
        return exploitation + exploration

    def select(self) -> ModelMCTSNode:
        """
        Traverse tree from root selecting child with highest UCT until a node with unexpanded children
        or leaf is found. Returns that node (the expansion point).
        """
        current = self.root
        while True:
            # If node is not fully expanded, return it
            if len(current.children) < len(self.vocab):
                return current
            # Otherwise choose child with max UCT score
            scores = [self.uct_score(c) for c in current.children]
            max_idx = int(scores.index(max(scores)))
            current = current.children[max_idx]

    def expand(self, node: ModelMCTSNode) -> Optional[ModelMCTSNode]:
        """
        Randomly expand node by adding a child with a command not yet in node.children.
        Returns the new child or None if no expansion possible.
        """
        available = [cmd for cmd in self.vocab if cmd not in [c.transformation for c in node.children]]
        if not available or node.depth >= self.max_recipe_len:
            return None
        cmd = random.choice(available)
        return node.add_child(cmd)

    def rollout(self, node: ModelMCTSNode) -> ModelMCTSNode:
        """
        Perform a random rollout (simulation) from node until max depth or no available commands.
        Returns the terminal node reached by the rollout.
        """
        current = node
        while current.depth < self.max_recipe_len:
            available = [cmd for cmd in self.vocab if cmd not in [c.transformation for c in current.children]]
            if not available:
                break
            cmd = random.choice(available)
            child = current.add_child(cmd)
            current = child
        return current

    def evaluate(self, node: ModelMCTSNode) -> ModelMCTSNode:
        """
        Use the predictive model to evaluate QoR for the recipe represented by `node`.
        Compute a reward (higher is better) and set node.power_value and node.reward.
        Reward is a sigmoid of the relative improvement from first_power to predicted power.
        """
        mean = self.global_mean
        std = self.global_std

        # Predict final QoR using helper (predict_design_qor imported from utils)
        pred_list = predict_design_qor(
            self.model,
            self.design,
            node.get_recipe_str(),
            std,
            mean,
            self.vocab,
            aigs_dir=self.aigs_dir,
            max_recipe_len=self.max_recipe_len,
            device=self.device
        )
        power = pred_list[-1]
        node.power_value = power

        # compute reward: normalized improvement
        diff = self.first_power - power
        norm = diff / self.first_power if self.first_power else 0.0
        sig = 1.0 / (1.0 + math.exp(-10.0 * norm))
        node.reward = float(sig)

        if self.best_node is None or (node.power_value is not None and node.power_value < getattr(self.best_node, "power_value", float("inf"))):
            self.best_node = node

        return node

    def backpropagate(self, node: ModelMCTSNode) -> None:
        """
        Backpropagate the node.reward up to the root, updating visits and cumulative rewards.
        """
        cur = node
        while cur:
            cur.visits += 1
            cur.reward += node.reward
            cur = cur.parent

    def run(self, iterations: int) -> Optional[ModelMCTSNode]:
        """
        Run MCTS for a number of iterations and return the best node found.
        """
        for _ in range(iterations):
            leaf = self.select()
            child = self.expand(leaf) or leaf
            leaf2 = self.rollout(child)
            evaluated = self.evaluate(leaf2)
            self.backpropagate(evaluated)
        return self.best_node
