import os
from huggingface_hub import InferenceClient

api_key = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACEHUB_API_TOKEN')
print('api_key present:', bool(api_key))
client = InferenceClient(api_key=api_key)
for model_id in ['gpt2', 'mistralai/Mistral-7B-Instruct-v0.3']:
    print('\n--- model', model_id, '---')
    try:
        print('model_info:', client.model_info(model_id).modelId)
    except Exception as e:
        print('model_info error:', type(e).__name__, e)
    try:
        out = client.text_generation('Hello, how are you?', model=model_id)
        print('text_generation ok:', type(out), out if isinstance(out, str) else 'non-str output')
    except Exception as e:
        print('text_generation error:', type(e).__name__, e)
    try:
        if hasattr(client, 'chat'):
            chat_fn = client.chat
            print('chat attr exists')
            try:
                out = chat_fn('Hello', model=model_id)
                print('chat ok:', type(out), out)
            except Exception as e:
                print('chat error:', type(e).__name__, e)
    except Exception as e:
        print('chat introspection error:', type(e).__name__, e)
