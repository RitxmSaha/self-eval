import multiprocessing
from mcts import MCTS, MCTSNode
from load_problems import format_dataset, Question
from typing import List, Tuple, Dict, Any
from llm import LLM
from tqdm import tqdm
import json
from datetime import datetime
import os

def process_question(model: LLM, question: Question) -> Tuple[str, List[float], List[Dict[str, float]]]:
    """Process a single question with token probability analysis using MCTS.
    
    Args:
        model: LLM instance
        question: Question object containing the problem
    Returns:
        Tuple of (generated text, chosen logprobs, token probabilities)
    """
    
    root_prompt = f"""Solve this math problem step by step. Separate each step with two newlines (\n\n). 
    After your final step, add two newlines and put just the final numerical answer in LaTeX boxed notation like this: \\boxed{{answer}}

    Here's the question:
    {question.question}

    Let me solve this step by step:"""

    root = MCTSNode(root_prompt, [], None, False, 100.0)

    mcts = MCTS(model, root)

    for _ in range(2):
        mcts.step()


    
    return mcts.export_tree()

def main():
    model = LLM("nvidia/OpenMath2-Llama3.1-8B")
    # Load AIME questions
    questions = format_dataset("qq8933/AIME_1983_2024")
    print(f"Loaded {len(questions)} AIME questions")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/AIME-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a pool of workers
    num_processes = 32

    questions = questions[0:1]
    
    # Process questions in parallel with progress bar
    with multiprocessing.Pool(num_processes) as pool:
        for idx, result in enumerate(tqdm(
            pool.starmap(process_question, [(model, question) for question in questions]),
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
