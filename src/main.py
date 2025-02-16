import multiprocessing
from datasets import load_aime_questions
from typing import List, Tuple, Dict, Any
from llm import LLM
from tqdm import tqdm

def process_question(args: Tuple[LLM, str]) -> Tuple[str, List[Dict[str, float]]]:
    """Process a single question with token probability analysis.
    
    Args:
        args: Tuple containing (LLM instance, question text)
    Returns:
        Tuple of (generated text, token probabilities)
    """
    model, question = args
    
    prompt = f"""Solve this math problem step by step. Separate each step with two newlines (\n\n). 
    After your final step, add two newlines and put just the final numerical answer in LaTeX boxed notation like this: \boxed{{answer}}

    Here's the question:
    {question}

    Let me solve this step by step:"""

    outputs = model.generate_with_probs(
        [prompt],
        temperature=0.95,
        max_tokens=512
    )
    return outputs[0]  # Return first (and only) result

def main():
    model = LLM("Qwen/Qwen2.5-Coder-32B-Instruct")
    # Load AIME questions
    questions = load_aime_questions()
    print(f"Loaded {len(questions)} AIME questions")
    
    # Create a pool of workers
    num_processes = 32
    
    # Prepare arguments for each worker
    args = [(model, question) for question in questions]
    
    # Process questions in parallel with progress bar
    with multiprocessing.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_question, args),
            total=len(questions),
            desc="Processing questions"
        ))
    
    # Process results
    for i, (text, token_data) in enumerate(results):
        print(f"\nQuestion {i+1}:")
        print(f"Generated text: {text}")
        print("\nToken-by-token alternatives:")
        for j, alternatives in enumerate(token_data, 1):
            print(f"\nToken {j} alternatives:")
            sorted_alternatives = sorted(alternatives.items(), key=lambda x: x[1], reverse=True)
            for token, prob in sorted_alternatives[:10]:  # Show top 10 alternatives
                print(f"'{token}': {prob:.4f}")

if __name__ == "__main__":
    main()
