import inspect
from langchain_huggingface import HuggingFaceEndpoint
print('HuggingFaceEndpoint signature:', inspect.signature(HuggingFaceEndpoint))
print('HuggingFaceEndpoint source file:', inspect.getsourcefile(HuggingFaceEndpoint))
source = inspect.getsource(HuggingFaceEndpoint)
print('\n--- SOURCE PREVIEW ---\n')
print(source[:3000])
