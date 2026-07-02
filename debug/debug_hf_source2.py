import inspect
from langchain_huggingface import HuggingFaceEndpoint
src = inspect.getsource(HuggingFaceEndpoint)
start = src.find('def _call')
end = src.find('\n\n', start)
print(src[start:end])
