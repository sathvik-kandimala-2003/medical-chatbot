import os
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFacePipeline

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def local_generate(prompt):
    generator = pipeline("text-generation", model="gpt2")
    result = generator(prompt, max_new_tokens=100)
    return result[0]['generated_text']


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
Context: {context}
Question: {question}
Answer the question using only the context above. Be clear and concise.
"""

        HUGGINGFACE_REPO_ID = "gpt2"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        print("HF_TOKEN:", HF_TOKEN)

        try: 
            vectorstore = get_vectorstore()
            print("Vectorstore:", vectorstore)
            if vectorstore is None:
                st.error("Failed to load the vector store")

            print("Loading LLM...")
            llm = load_llm()
            print("LLM loaded:", llm)

            # Create the PromptTemplate object
            custom_prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

            print("Building QA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
                chain_type_kwargs={"prompt": custom_prompt_template}
            )
            print("QA chain built. Running RAG inference...")
# this is to get summary...it is too small
            # from transformers import pipeline
            # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            # summary = summarizer(result_to_show, max_length=60, min_length=20, do_sample=False)
            # result_to_show = summary[0]['summary_text']

            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(prompt)
            print("Retrieved docs:", docs)
            print("First doc content:", docs[0].page_content if docs else "None")

            if not docs:
                result_to_show = "Sorry, I couldn't find any relevant information in the document."
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role':'assistant', 'content': result_to_show})
            else:
                try:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type_kwargs={"prompt": custom_prompt_template}
                    )
                    response = qa_chain({"query": prompt})
                    result_to_show = response["result"]
                    print("Raw LLM output:", result_to_show)

                    # Post-process to remove everything before the answer instruction
                    split_key = "You are a medical expert."
                    if split_key in result_to_show:
                        result_to_show = result_to_show.split(split_key, 1)[-1].strip()

                    # Further filter out lines that repeat instructions or context
                    lines = result_to_show.splitlines()
                    filtered_lines = [
                        line for line in lines
                        if not line.strip().lower().startswith(("context:", "question:", "use the pieces", "start the answer"))
                        and line.strip() != ""
                    ]
                    result_to_show = "\n".join(filtered_lines)

                    # Summarize if the answer is long enough
                    if len(result_to_show.split()) > 120:  # Only summarize if input is long
                        from transformers import pipeline
                        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                        summary = summarizer(result_to_show, max_length=180, min_length=100, do_sample=False)
                        result_to_show = summary[0]['summary_text']

                    st.chat_message('assistant').markdown(result_to_show)
                    st.session_state.messages.append({'role':'assistant', 'content': result_to_show})
                except Exception as e:
                    print("Exception in QA chain:", e)
                    result_to_show = "Sorry, I couldn't generate an answer for your question."
                    st.chat_message('assistant').markdown(result_to_show)
                    st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            import traceback
            print("Exception:", e)
            traceback.print_exc()
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()