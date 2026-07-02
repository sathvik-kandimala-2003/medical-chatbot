from huggingface_hub import InferenceClient
client = InferenceClient(token=None)
model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
print('model id', model_id)
try:
    info = client.model_info(model_id)
    print('model info loaded')
    print('name', info.modelId if hasattr(info, 'modelId') else info.modelId)
except Exception as e:
    print('model_info error', type(e).__name__, e)
