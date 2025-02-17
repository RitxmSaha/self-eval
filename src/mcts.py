from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
from llm import LLM
from tenacity import retry, stop_after_attempt, wait_exponential

class MCTSNode:
    def __init__(self, text: str, probs: List[Dict[str, float]], parent: Optional['MCTSNode'] = None):
        self.text = text
        self.probs = probs
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 1
        self.value = 0.0

    def ucb_score(self, exploration_constant: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MCTS:
    def __init__(self, llm: LLM, root: MCTSNode):
        self.llm = llm
        self.root = root
        self.nodes = [root]
        self.exploration_constant = math.sqrt(2)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def step(self) -> Tuple[str, List[Dict[str, float]]]:
        """Generate a solution using MCTS search.
        
        Args:
            question: The math question to solve
            n_iterations: Number of MCTS iterations
            
        Returns:
            Tuple of (generated text, token probabilities)
        """
        # prompt = f"""Solve this math problem step by step. Separate each step with two newlines (\n\n). 
        # After your final step, add two newlines and put just the final numerical answer in LaTeX boxed notation like this: \boxed{{answer}}

        # Here's the question:
        # {question}

        # Let me solve this step by step:"""

        selected = self.select_ucb()
        prompt = self.build_prompt(selected)




        # Get initial solution attempt
        solution, probs = self.llm.generate_with_probs(
            prompt,
            temperature=0.95,
            max_tokens=512
        )[0]
        
        return solution, probs

    def build_prompt(self, node: MCTSNode) -> str:
        """Build a prompt by concatenating steps from root to current node.
        
        Args:
            node: Current MCTSNode to build prompt from
            
        Returns:
            String containing concatenated steps from root to current node
        """
        steps = []
        current = node
        
        # Traverse up the tree to collect all steps
        while current is not None:
            if current.state.current_step:  # Only add if there's a step
                steps.append(current.state.current_step)
            current = current.parent
        
        # Reverse steps to get them in correct order (root to leaf)
        steps.reverse()
        
        # Join steps with double newlines
        return "\n\n".join(steps)

    def select_ucb(self) -> MCTSNode:
        """Select a node using the UCB algorithm by traversing down the tree.
        
        Returns:
            Selected leaf node after traversing down the tree
        """
        current = self.root
        while current.children:  # While not at a leaf node
            current = max(current.children, key=lambda child: child.ucb)
        return current

    def back_propogate(self, node: MCTSNode):
        """Update UCB scores for all affected nodes after a simulation.
        
        Args:
            node: The leaf node from which to start backpropagation
        """
        current = node
        while current.parent is not None:
            # Update UCB scores for all siblings (including the current node)
            for sibling in current.parent.children:
                sibling.ucb_score(self.exploration_constant)
            current = current.parent


