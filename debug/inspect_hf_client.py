import inspect
from huggingface_hub import InferenceClient
print('InferenceClient methods:')
for name in sorted([name for name in dir(InferenceClient) if not name.startswith('_')]):
    print(name)
print('\ntext_generation signature:')
print(inspect.signature(InferenceClient.text_generation))
print('\nchat signature:')
if hasattr(InferenceClient, 'chat'):
    print(inspect.signature(InferenceClient.chat))
else:
    print('no chat method')
