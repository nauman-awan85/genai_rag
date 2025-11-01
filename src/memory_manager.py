import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

memory_db_path = "models/embeddings/memory_index"

# Embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create an empty FAISS index
def _create_empty_faiss_index(dim: int = 384):
    index = faiss.IndexFlatL2(dim)
    return FAISS(embedding_function=embedder, index=index, docstore={}, index_to_docstore_id={})

# Load or create the memory database
def _load_memory_db():
    if os.path.exists(memory_db_path):
        try:
            db = FAISS.load_local(memory_db_path, embedder, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            print(f"Failed to load memory DB: {e}, recreating.")
    return _create_empty_faiss_index()

# Save a query–answer pair to memory
def save_to_memory(query: str, answer: str):
    db = _load_memory_db()
    text = f"Q: {query}\nA: {answer}"
    db.add_texts([text])
    db.save_local(memory_db_path)
    print("Memory updated.")

# Retrieve the most relevant past memories
def retrieve_memory_context(query: str, k: int = 3) -> str:
    db = _load_memory_db()
    results = db.similarity_search(query, k=k)
    if not results:
        return ""
    return "\n".join([r.page_content for r in results])



# https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html
# https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html