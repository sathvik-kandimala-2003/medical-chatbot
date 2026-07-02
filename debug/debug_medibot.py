import os, traceback
from medibot import get_vectorstore, load_llm, DB_FAISS_PATH

print('cwd:', os.getcwd())
print('db path exists:', os.path.exists(DB_FAISS_PATH))
print('db full path:', os.path.abspath(DB_FAISS_PATH))

try:
    db = get_vectorstore()
    print('vectorstore loaded:', type(db))
except Exception as e:
    print('get_vectorstore failed:', type(e).__name__, str(e))
    traceback.print_exc()

try:
    llm = load_llm()
    print('llm loaded:', type(llm))
except Exception as e:
    print('load_llm failed:', type(e).__name__, str(e))
    traceback.print_exc()
