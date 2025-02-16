from litellm import completion
import os
import numpy as np

api_key = os.getenv("CUSTOM_LLM_API_KEY")

try:
    response = completion(
        model="openai/Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.95,
        timeout=30,
        api_base="http://avior.mlfoundry.com/live-inference/v1",
        api_key=api_key,
        logprobs=True,
        top_p=0.95
    )
    
    print("Response received!")
    print(f"Content: {response.choices[0].message.content}")
    print("\nToken Probabilities:")
    
    # Debug print to see the structure
    logprobs = response.choices[0].logprobs
    print(f"Logprobs type: {type(logprobs)}")
    print(f"Logprobs content: {logprobs}")
    
    # Handle tuple structure if present
    if isinstance(logprobs, tuple):
        token, logprob = logprobs
        prob = np.exp(logprob)
        print(f"Token: '{token}', Probability: {prob:.4f}\n")
    elif isinstance(logprobs, list):
        for token_info in logprobs:
            if isinstance(token_info, tuple):
                token, logprob = token_info
                prob = np.exp(logprob)
                print(f"Token: '{token}', Probability: {prob:.4f}\n")
    
    print(f"\nFull response: {response}")

except Exception as e:
    print(f"Error: {str(e)}")