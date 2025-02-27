from litellm import completion
import math
import os


# Define the API base URL and model name
api_base = "http://avior.mlfoundry.com/live-inference/v1"
api_key = os.getenv("CUSTOM_LLM_API_KEY")
model_name = "openai/nvidia/OpenMath2-Llama3.1-8B"

# Prepare the message for the model
messages = [{"role": "user", "content": "What is the capital of France?"}]

# Make the completion call
response = completion(
    model=model_name,
    messages=messages,
    api_base=api_base,
    api_key=api_key,
    logprobs=True,
    top_logprobs=10,
    max_tokens=50
)

# Print the response content
print(response['choices'][0]['message']['content'])
choice_logprobs = response['choices'][0]['logprobs']

#print(choice_logprobs)
#exit()


for i, token_logprob in enumerate(choice_logprobs['content']):
    token = token_logprob.token
    logprob = token_logprob.logprob
    top_logprobs = token_logprob.top_logprobs

    print(f"Token {i + 1}: '{token}' (Log Probability: {logprob})")
    print("Top potential tokens and their probabilities:")

    for top in top_logprobs:
        top_token = top.token
        top_logprob = top.logprob
        probability = math.exp(top_logprob)  # Convert log probability to probability
        print(f"  Token: '{top_token}', Probability: {probability:.4f}")

    print("\n" + "-"*50 + "\n")