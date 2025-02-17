import multiprocessing
from load_problems import format_dataset, Question
from typing import List, Tuple, Dict, Any
from llm import LLM
from tqdm import tqdm

def process_question(args: Tuple[LLM, Question]) -> Tuple[str, List[float], List[Dict[str, float]]]:
    """Process a single question with token probability analysis.
    
    Args:
        args: Tuple containing (LLM instance, question text)
    Returns:
        Tuple of (generated text, chosen logprobs, token probabilities)
    """
    model, question = args
    
    prompt = f"""Solve this math problem step by step. Separate each step with two newlines (\n\n). 
    After your final step, add two newlines and put just the final numerical answer in LaTeX boxed notation like this: \boxed{{answer}}

    Here's the question:
    {question}

    Let me solve this step by step:"""

    # generate_with_probs now returns (text, logprobs, token_probs)
    text, logprobs, token_probs = model.generate_with_probs(
        prompt,  # Now just passing a single prompt string
        temperature=0.95,
        max_tokens=512
    )
    
    return text, logprobs, token_probs

def main():
    model = LLM("nvidia/OpenMath2-Llama3.1-8B")
    # Load AIME questions
    questions = format_dataset("qq8933/AIME_1983_2024")
    print(f"Loaded {len(questions)} AIME questions")
    
    # Create a pool of workers
    num_processes = 32
    
    # Prepare arguments for each worker
    args = [(model, question) for question in questions]
    args = args[100:101]
    
    # Process questions in parallel with progress bar
    with multiprocessing.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_question, args),
            total=len(questions),
            desc="Processing questions"
        ))
    
    # Process results
    for i, (text, logprobs, token_data) in enumerate(results):
        print(f"\nQuestion {i+1}:")
        print(f"Generated text: {text}")
        print("\nToken-by-token alternatives:")
        for j, alternatives in enumerate(token_data, 1):
            print(f"\nToken {j} alternatives:")
            sorted_alternatives = sorted(alternatives.items(), key=lambda x: x[1], reverse=True)
            for token, prob in sorted_alternatives:  # Show top 10 alternatives
                print(f"'{token}': {prob:.4f}")

if __name__ == "__main__":
    main()
