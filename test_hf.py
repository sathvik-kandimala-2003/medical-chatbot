from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

# Load variables from .env
load_dotenv()

# Read Hugging Face token from .env
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

try:
    result = client.text_generation(
        "Hello, how are you?",
        model="gpt2"
    )
    print(result)
except Exception as e:
    print("Error:", e)
