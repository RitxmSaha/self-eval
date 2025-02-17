import multiprocessing
from typing import List, Tuple, Dict, Any
import os
import numpy as np
from litellm import completion

class LLM:
    def __init__(self, model_path: str):
        """Initialize LLM with the specified model.
        
        Args:
            model_path: Path or name of the model to load
        """
        self.model_name = f"openai/{model_path}"  # Add openai/ prefix for litellm
        self.api_key = os.getenv("CUSTOM_LLM_API_KEY")
        self.api_base = "http://avior.mlfoundry.com/live-inference/v1"
        
    def generate_with_probs(
        self,
        prompt: str,
        temperature: float = 0.95,
        max_tokens: int = 512,
    ) -> Tuple[str, List[Tuple[str, float]], List[Dict[str, float]]]:
        """Generate completion with token probabilities for a given prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple containing:
                - Generated text
                - List of tuples (token, logprob) for chosen tokens
                - List of dictionaries containing token alternatives and their probabilities
        """
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
            api_base=self.api_base,
            api_key=self.api_key,
            logprobs=True,
            top_logprobs=20,
            top_p=0.95
        )
        
        # Extract generated text and token probabilities
        generated_text = response.choices[0].message.content
        
        # Process token probabilities with alternatives
        token_data = []
        chosen_logprobs = []
        try:
            logprobs_data = response.choices[0].logprobs.content
            
            for token_info in logprobs_data:
                token_alternatives = {}
                # Store token and logprob as tuple
                chosen_logprobs.append((token_info.token, token_info.logprob))
                
                # Add the main token and alternatives in one go
                all_alternatives = [
                    (token_info.token, token_info.logprob)
                ] + [(alt.token, alt.logprob) for alt in token_info.top_logprobs]
                
                # Convert logprobs to probabilities directly
                for token, logprob in all_alternatives:
                    if logprob is not None:
                        token_alternatives[token] = np.exp(logprob)
                        
                token_data.append(token_alternatives)
                
        except Exception as e:
            print(f"Error processing logprobs: {str(e)}")
        
        return generated_text, chosen_logprobs, token_data

def process_single_question(args: Tuple[LLM, str]) -> Tuple[str, List[Dict[str, float]]]:
    """Process a single AIME question with token probability analysis.
    
    Args:
        args: Tuple containing (LLM instance, question text)
    Returns:
        Tuple of (generated text, token probabilities)
    """
    model, question = args
    outputs = model.generate_with_probs(
        question,
        temperature=0.95,
        max_tokens=512
    )
    return outputs[0], outputs[2]  # Return first (and only) result and token probabilities