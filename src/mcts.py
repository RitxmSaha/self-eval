from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
from llm import LLM

class MCTSNode:
    def __init__(self, text: str, probs: List[Dict[str, float]], parent: Optional['MCTSNode'] = None, terminal: Optional[bool] = False, value: Optional[float] = 0.0):
        self.text = text
        self.probs = probs
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.terminal = terminal
        self.visits = 1
        self.value = value
        self.value_reasoning = []

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
        
    def step(self) -> Tuple[str, List[Dict[str, float]]]:
        """Generate a solution using MCTS search.
        
        Args:
            question: The math question to solve
            n_iterations: Number of MCTS iterations
            
        Returns:
            Tuple of (generated text, token probabilities)
        """

        selected = self.select_ucb()
        prompt = self.build_prompt(selected)

        # Get initial solution attempt
        solution, solution_probs, total_probs = self.llm.generate_with_probs(
            prompt,
            temperature=0.95,
            max_tokens=512
        )

        steps = solution.split('\n\n')

        step_tuples = []
        
        idx = 0
        for step in steps:
            accumulated_len = 0
            probs = []
            while accumulated_len < len(step):
                probs.append(total_probs[idx])
                accumulated_len += len(solution_probs[0])
                idx += 1

            while(solution_probs[idx][0] ==  "\n\n"):
                idx += 1

            step_tuples.append((step, probs))

        added_nodes = []
        for step, probs in step_tuples:
            node_parent = selected if not added_nodes else added_nodes[-1]
            new_node = MCTSNode(step, probs, parent=node_parent, terminal=False)
            self.assign_reward(new_node)
            added_nodes.append(new_node)
            node_parent.children.append(new_node)

        added_nodes[-1].terminal = True
        self.nodes.extend(added_nodes)

        ### VISITING LOGIC START ###
        for i, node in enumerate(added_nodes.reverse(), start=1):
            node.parent.visits += i
            self.assign_reward(node)

        parent_traverse = selected
        while parent_traverse.parent is not None:
            parent_traverse.parent.visits += len(added_nodes)
            parent_traverse = parent_traverse.parent
        ### VISITING LOGIC END ###

        self.back_propogate(added_nodes[-1])
        return

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
        while [child for child in current.children if not child.terminal]:
            current = max([child for child in current.children if not child.terminal],
                           key=lambda child: child.ucb)
        return current
    
    def assign_reward(self, node: MCTSNode):
        """Assign a reward to a node based on the final solution using LLM evaluation.
        
        Args:
            node: The leaf node to assign a reward to
        """
        original_trajectory = self.build_prompt(node.parent)

        evaluation_prompt = f"""
        Context (Previous Steps):
        {original_trajectory}

        Step to Evaluate:
        {node.text}

        Evaluate ONLY the newest step shown above under "Step to Evaluate". The previous steps are provided only for context.
        
        Strictly critic and analyze this step. Point out any logical flaws in the reasoning process under [Analysis]. Output a score between -100 to +100 that represents the quality of this specific step under [Score], using LaTeX boxed notation.

        Use the following rubric to assign scores:
        +75 to +100:
        The step contains no logical errors and builds appropriately on previous steps.
        +0 to +74:
        The step contains minor logical errors but generally moves in the right direction.
        -74 to -1:
        The step contains major logical errors but shows some understanding.
        -100 to -75:
        The step contains fundamental misunderstandings or completely incorrect reasoning.

        Response format:
        [Analysis]...
        [Score]\\boxed{{score}}"""

        score_text = self.llm.generate(
            evaluation_prompt,
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=512    # Increased to accommodate analysis + score
        ).strip()
        
        try:
            # Find the last instance of score in \boxed{...} format
            start_idx = score_text.rfind("\\boxed{") + 7  # +7 to skip "\boxed{"
            end_idx = score_text.find("}", start_idx)
            score_str = score_text[start_idx:end_idx]
            
            score = float(score_str)
            # Ensure score is within valid range
            score = max(-100.0, min(100.0, score))  # Updated range to -100 to 100
            node.value = score
            node.value_reasoning.append(score_text)
        except (ValueError, IndexError):
            # Fallback value if LLM doesn't return a valid number or format
            node.value = 0.0
            node.value_reasoning.append(score_text)


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

    def export_tree(self) -> List[Tuple[str, float, List[str]]]:
        """Export the MCTS tree as a nested dictionary structure.
        
        Returns:
            A dictionary containing the entire tree structure with all node information
        """
        def node_to_dict(node: MCTSNode) -> dict:
            return {
                'text': node.text,
                'probs': node.probs,
                'terminal': node.terminal,
                'visits': node.visits,
                'value': node.value,
                'value_reasoning': node.value_reasoning,
                'children': [node_to_dict(child) for child in node.children]
            }
        
        return node_to_dict(self.root)

        
