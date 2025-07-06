from huggingface_hub import InferenceClient

client = InferenceClient(token="enter your hugging face token here")
try:
    result = client.text_generation("Hello, how are you?", model="mistralai/Mistral-7B-Instruct-v0.3")
    print(result)
except Exception as e:
    print("Error:", e)