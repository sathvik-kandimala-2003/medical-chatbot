import traceback
from medibot import get_vectorstore, load_llm, retrieve_documents, set_custom_prompt, CUSTOM_PROMPT_TEMPLATE
from langchain_classic.chains import RetrievalQA

prompt = "how is diabetes cured?"
print('Prompt:', prompt)
try:
    vectorstore = get_vectorstore()
    print('vectorstore type', type(vectorstore))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retrieve_documents(retriever, prompt)
    print('docs len', len(docs))
    for i, doc in enumerate(docs[:3], 1):
        print('doc', i, repr(doc.page_content[:200]))
except Exception as e:
    print('retrieve_documents failed', type(e).__name__, str(e))
    traceback.print_exc()

try:
    llm = load_llm()
    print('llm type', type(llm))
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )
    response = qa_chain.invoke({"query": prompt})
    print('response', response)
except Exception as e:
    print('qa_chain invoke failed', type(e).__name__, str(e))
    traceback.print_exc()
