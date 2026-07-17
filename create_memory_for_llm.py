import logging

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from retrieval import BM25Index
from retrieval.config import BM25_INDEX_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# 1000/200 keeps a full symptom/cause/treatment paragraph together far more
# often than the previous 500/50, without making chunks so large that
# retrieval loses precision.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(extracted_data)


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def main():
    logger.info("Loading PDFs from %s", DATA_PATH)
    documents = load_pdf_files(DATA_PATH)

    logger.info("Splitting %d page(s) into chunks (size=%d, overlap=%d)", len(documents), CHUNK_SIZE, CHUNK_OVERLAP)
    text_chunks = create_chunks(documents)

    embedding_model = get_embedding_model()

    logger.info("Building FAISS index from %d chunk(s)", len(text_chunks))
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)

    # BM25 is built from the exact same chunks as FAISS so hybrid retrieval
    # fuses two views of one corpus, not two different corpora.
    bm25_index = BM25Index.build(text_chunks)
    bm25_index.save(BM25_INDEX_PATH)

    print(f"Indexed {len(text_chunks)} chunks from {len(documents)} pages into {DB_FAISS_PATH}")
    print(f"Saved BM25 keyword index to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
