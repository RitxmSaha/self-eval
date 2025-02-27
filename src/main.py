import multiprocessing
from mcts import MCTS, MCTSNode
from load_problems import format_dataset, Question
from typing import List, Tuple, Dict, Any
from llm import LLM
from tqdm import tqdm
import json
from datetime import datetime
import os

def process_question(model_str: str, question: Question) -> Tuple[str, List[float], List[Dict[str, float]]]:
    """Process a single question with token probability analysis using MCTS.
    
    Args:
        model: LLM instance
        question: Question object containing the problem
    Returns:
        Tuple of (generated text, chosen logprobs, token probabilities)
    """
    
    root_prompt = f"""Solve this math problem step by step.

Instructions:
1. Break down your solution into clear, logical steps
2. Separate each step with two newlines (\n\n)
3. Show all your work and calculations
4. After your final step, add two newlines
5. End with ONLY the final numerical answer in LaTeX boxed notation: \\boxed{{answer}}

Example of expected format:
Step 1: Understand the problem. We need to find the area of a triangle with sides 5, 12, and 13.
\n\n
Step 2: First, I'll check if this is a right triangle by using the Pythagorean theorem. 5² + 12² = 25 + 144 = 169 = 13². This is a right triangle with legs 5 and 12, and hypotenuse 13.
\n\n
Step 3: For a right triangle, the area can be calculated as (base × height)/2. Using the legs as base and height: Area = (5 × 12)/2 = 60/2 = 30.
\n\n
\\boxed{30}

Here's the question:
{question.question}

Let me solve this step by step:"""

    model = LLM(model_str)

    root = MCTSNode(root_prompt, [], None, False, 100.0)

    mcts = MCTS(model, root)

    for _ in range(25):
        mcts.step()


    
    return mcts.export_tree(question)

def main():
    model_str = "nvidia/OpenMath2-Llama3.1-8B"
    # Load AIME questions
    questions = format_dataset("qq8933/AIME_1983_2024")
    print(f"Loaded {len(questions)} AIME questions")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/AIME-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a pool of workers
    num_processes = 32

    questions = questions[0:5]
    
    # Process questions in parallel with progress bar
    with multiprocessing.Pool(num_processes) as pool:
        for idx, result in enumerate(tqdm(
            pool.starmap(process_question, [(model_str, question) for question in questions]),
            total=len(questions),
            desc="Processing questions"
        )):
            # Save each result to its own JSON file
            output_path = os.path.join(output_dir, f"{idx+1}.json")
            with open(output_path, 'w') as f:
                json.dump({
                    'question_index': idx,
                    'question_text': questions[idx].question,
                    'question_answer': questions[idx].answer,
                    'result': result
                }, f, indent=2)

if __name__ == "__main__":
    main()
