import re
from typing import Optional

def extract_number(text: str) -> Optional[str]:
    """Extract the last number from text, handling boxed LaTeX notation."""
    # Handle LaTeX \boxed{} notation
    if "\\boxed" in text:
        box_match = re.search(r'\\boxed{([^}]+)}', text)
        if box_match:
            text = box_match.group(1)
    
    # Find all numbers in the text
    numbers = re.findall(r'-?\d*\.?\d+', text.replace(",", ""))
    return numbers[-1] if numbers else None

def verify_math_answer(prediction: str, ground_truth: str, tolerance: float = 1e-5) -> bool:
    """
    Verify if a predicted answer matches the ground truth.
    
    Args:
        prediction: The predicted answer string
        ground_truth: The correct answer string
        tolerance: Floating point comparison tolerance
        
    Returns:
        bool: Whether the prediction matches the ground truth
    """
    # Extract numbers from both strings
    pred_num = extract_number(prediction)
    true_num = extract_number(ground_truth)
    
    if pred_num is None or true_num is None:
        return False
        
    try:
        # Convert to floats and compare
        pred_val = float(pred_num)
        true_val = float(true_num)
        return abs(pred_val - true_val) < tolerance
    except ValueError:
        # If conversion fails, do exact string match
        return pred_num == true_num
