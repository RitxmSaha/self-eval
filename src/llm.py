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
        prompts: List[str],
        temperature: float = 0.95,
        max_tokens: int = 512,
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Generate completions with token probabilities for given prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            List of tuples containing:
                - Generated text
                - List of dictionaries containing token alternatives and their probabilities
        """
        results = []
        
        for prompt in prompts:
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
            try:
                logprobs_data = response.choices[0].logprobs.content
                
                for token_info in logprobs_data:
                    token_alternatives = {}
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
            
            results.append((generated_text, token_data))
            
        return results

# Example usage:
if __name__ == "__main__":
    model = LLM("nvidia/OpenMath2-Llama3.1-8B")
    
    prompts = ["What is quantum physics?"]
    try:
        outputs = model.generate_with_probs(
            prompts,
            temperature=0.95,
            max_tokens=50
        )
        
        for text, token_data in outputs:
            print(f"\nGenerated text: {text}")
            print("\nToken-by-token alternatives:")
            for i, alternatives in enumerate(token_data, 1):
                print(f"\nToken {i} alternatives:")
                # Sort alternatives by probability
                sorted_alternatives = sorted(alternatives.items(), key=lambda x: x[1], reverse=True)
                for token, prob in sorted_alternatives:
                    print(f"'{token}': {prob:.4f}")
    except Exception as e:
        print(f"Error running inference: {str(e)}") 