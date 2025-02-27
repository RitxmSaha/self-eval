from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
from llm import LLM
from load_problems import Question

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
        self.ucb = 0.0

    def ucb_score(self, exploration_constant: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float('inf')
        #exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploration

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
            max_tokens=4096
        )

        # Simple but robust step parsing
        steps = solution.split('\n\n')
        step_tuples = []
        
        # Check the structure of solution_probs
        if solution_probs and isinstance(solution_probs[0], tuple):
            # If solution_probs is a list of tuples, extract the strings
            tokens = [token for token, _ in solution_probs]
        else:
            # Otherwise assume it's already a list of strings
            tokens = solution_probs
        
        # Build the full text from tokens for reference
        full_text = "".join(tokens)
        
        # Track position in both the solution text and token list
        text_pos = 0
        token_pos = 0
        
        for step in steps:
            if not step.strip():  # Skip empty steps
                continue
                
            step_probs = []
            step_end_pos = text_pos + len(step)
            
            # Collect tokens until we've covered this step
            current_text_pos = text_pos
            while token_pos < len(tokens) and current_text_pos < step_end_pos:
                token = tokens[token_pos]
                step_probs.append(total_probs[token_pos])
                current_text_pos += len(token)
                token_pos += 1
            
            # Update text position for next step
            text_pos = step_end_pos
            
            # Skip any "\n\n" tokens between steps
            while token_pos < len(tokens) and "\n" in tokens[token_pos]:
                text_pos += len(tokens[token_pos])
                token_pos += 1
                
            step_tuples.append((step, step_probs))

        # Debug: Print steps and their average entropy
        print("\n===== STEP PARSING DEBUG =====")
        for i, (step, probs) in enumerate(step_tuples):
            # Calculate entropy for each token's probability distribution
            step_entropies = []
            for prob_dist in probs:
                # Filter out zero probabilities
                valid_probs = {k: v for k, v in prob_dist.items() if v > 0}
                if valid_probs:
                    # Calculate entropy: -sum(p * log(p))
                    entropy = -sum(p * math.log(p, 2) for p in valid_probs.values())
                    step_entropies.append(entropy)
            
            # Calculate average entropy for this step
            avg_entropy = sum(step_entropies) / len(step_entropies) if step_entropies else 0
            
            # Print step info (truncate long steps for readability)
            max_display_len = 50
            display_step = step[:max_display_len] + "..." if len(step) > max_display_len else step
            display_step = display_step.replace('\n', '\\n')  # Make newlines visible
            
            print(f"Step {i+1}: '{display_step}'")
            print(f"  Tokens: {len(probs)}, Avg Entropy: {avg_entropy:.2f}")
        
        print("=============================\n")

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
        for i, node in enumerate(reversed(added_nodes), start=1):
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
            steps.append(current.text)  # Use text directly from the node
            current = current.parent
        
        # Reverse steps to get them in correct order (root to leaf)
        steps.reverse()
        
        # Join steps with double newlines
        return "\n\n".join(steps) + "\n\n"

    # def select_ucb(self) -> MCTSNode:
    #     """Select a node using the UCB algorithm with progressive widening.
        
    #     Progressive widening controls the number of children that can be created
    #     for a node based on the number of visits:
    #     max_children = ceil(C * N^alpha)
    #     where:
    #     - C is the widening constant
    #     - N is the number of visits to the parent
    #     - alpha is the widening exponent (typically 0.25 to 0.5)
        
    #     Returns:
    #         Selected leaf node after traversing down the tree
    #     """
    #     current = self.root
    #     C = 1.0  # Widening constant
    #     alpha = 0.3  # Widening exponent
        
    #     while True:
    #         # Calculate maximum allowed children for current node
    #         max_children = math.ceil(C * (current.visits ** alpha))
            
    #         # If we have fewer children than allowed by progressive widening,
    #         # this node is eligible for expansion
    #         if len(current.children) < max_children:
    #             return current
                
    #         # Otherwise, continue down the tree using UCB
    #         non_terminal_children = [child for child in current.children if not child.terminal]
    #         if not non_terminal_children:
    #             return current
                
    #         current = max(non_terminal_children, key=lambda child: child.ucb)

    def select_ucb(self) -> MCTSNode:
        """Select a node using breadth-first exploration.
        
        This implementation ensures that each node gets exactly 3 children
        before moving on to explore the next level of the tree. This creates
        an evenly balanced tree where all nodes at the same level have the
        same number of children before deeper exploration begins.
        
        Returns:
            Selected node for expansion
        """
        # Start BFS from the root
        queue = [self.root]
        
        while queue:
            current = queue[0]
            
            # Skip terminal nodes
            if current.terminal:
                queue.pop(0)
                continue
                
            # If this node has fewer than 3 children, select it for expansion
            if len(current.children) < 3:
                return current
                
            # Otherwise, add its children to the queue and move to the next node
            queue.extend([child for child in current.children if not child.terminal])
            queue.pop(0)
            
        # If we've exhausted the queue, return the root (should rarely happen)
        return self.root
    
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

        # score_text, _, _ = self.llm.generate_with_probs(
        #     evaluation_prompt,
        #     temperature=0.1,  # Low temperature for consistent scoring
        #     max_tokens=512    # Increased to accommodate analysis + score
        # )

        #score_text = score_text.strip()
        score_text = "0"
        
        try:
            # Find the last instance of score in \boxed{...} format
            #start_idx = score_text.rfind("\\boxed{") + 7  # +7 to skip "\boxed{"
            #end_idx = score_text.find("}", start_idx)
            #score_str = score_text[start_idx:end_idx]
            
            #score = float(score_str)
            score = 0.0
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

    def export_tree(self, question: Question) -> dict:
        """Export the MCTS tree as a nested dictionary structure.
        
        Returns:
            A dictionary containing the entire tree structure with all node information
        """
        def node_to_dict(node: MCTSNode) -> dict:
            result = {
                'text': node.text,
                'probs': node.probs,
                'terminal': node.terminal,
                'visits': node.visits,
                'value': node.value,
                'value_reasoning': node.value_reasoning,
                'answer_text': None,
                'answer_correct': None,
                'children': [node_to_dict(child) for child in node.children]
            }
            
            # For terminal nodes, check the answer against the question verifier
            if node.terminal:
                answer_text = node.text
                result['answer_text'] = answer_text
                
                # Verify the answer if a verifier is available
                if hasattr(question, 'verifier') and question.verifier is not None:
                    # Pass both the answer and the ground truth to the verifier
                    if hasattr(question, 'answer'):
                        result['answer_correct'] = question.verifier(answer_text, question.answer)
                    else:
                        # If no ground truth is available, we can't verify
                        result['answer_correct'] = None
            
            return result
        
        return node_to_dict(self.root)

        
