from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
from llm import LLM
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class MCTSState:
    """Represents a state in the solution process"""
    steps: List[str]  # Steps taken so far
    current_step: str  # Current step being considered
    final_answer: Optional[str] = None  # Final boxed answer if terminal
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (has final answer)"""
        return self.final_answer is not None

class MCTSNode:
    def __init__(self, state: MCTSState, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        
    def add_child(self, state: MCTSState) -> 'MCTSNode':
        child = MCTSNode(state, parent=self)
        self.children.append(child)
        return child
    
    def ucb_score(self, exploration_constant: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MCTS:
    def __init__(self, llm: LLM):
        self.llm = llm
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, question: str, n_iterations: int = 100) -> Tuple[str, List[Dict[str, float]]]:
        """Generate a solution using MCTS search.
        
        Args:
            question: The math question to solve
            n_iterations: Number of MCTS iterations
            
        Returns:
            Tuple of (generated text, token probabilities)
        """
        prompt = f"""Solve this math problem step by step. Separate each step with two newlines (\n\n). 
        After your final step, add two newlines and put just the final numerical answer in LaTeX boxed notation like this: \boxed{{answer}}

        Here's the question:
        {question}

        Let me solve this step by step:"""

        # Get initial solution attempt
        solution, probs = self.llm.generate_with_probs(
            [prompt],
            temperature=0.95,
            max_tokens=512
        )[0]
        
        return solution, probs