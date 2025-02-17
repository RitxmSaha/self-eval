from datasets import load_dataset
from typing import List, Callable
import re
from verify_math import verify_math_answer

class Question:
    def __init__(self, question: str, answer: str, verifier: Callable[[str, str], bool]):
        self.question = question
        self.answer = answer
        self.verifier = verifier

    def verify(self, prediction: str) -> bool:
        return self.verifier(prediction, self.answer)

def verify_aime_answer(prediction: str, ground_truth: str) -> bool:
    """
    Verify AIME answer. AIME problems always have integer answers 
    between 0 and 999 inclusive.
    """
    try:
        # Extract the last number from both prediction and ground truth
        numbers = re.findall(r'\d+', prediction)
        pred_num = int(numbers[-1]) if numbers else None
        
        true_num = int(ground_truth)
        
        # AIME answers are always integers between 0 and 999
        if pred_num is None or pred_num < 0 or pred_num > 999:
            return False
            
        return pred_num == true_num
    except:
        return False

def format_dataset(dataset_name: str, split: str = "train") -> List[Question]:
    """
    Load and format a dataset from HuggingFace hub into a list of Questions.
    Automatically determines the appropriate verifier based on the dataset name.
    """
    dataset = load_dataset(dataset_name, split=split)
    
    # Determine verifier based on dataset name
    if "AIME" in dataset_name.upper():
        verifier = verify_aime_answer
    else:
        verifier = verify_math_answer
    
    questions = []
    for item in dataset:
        question = Question(
            question=item["Question"],
            answer=item["Answer"],
            verifier=verifier
        )
        questions.append(question)
    
    return questions