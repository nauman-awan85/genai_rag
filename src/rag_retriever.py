import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

index_path = "models/embeddings/faiss_index"


# create faiss vector store from documents in data/docs
def create_vectorstore(data_dir="data/docs"):
    docs = []

# load pdf and text documents
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)

        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif file.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

# split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

# create embeddings and build faiss index
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedder)

    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)

    print(f"indexed {len(chunks)} text chunks from {len(docs)} documents.")


# retrieve top-k relevant text chunks for a given query
def retrieve_context(query, k: int = 3):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)

    results = db.similarity_search_with_score(query, k=k)

    if not results:
        return "no relevant context found"

    seen = set()
    context_blocks = []

    for doc, score in results:
        source = os.path.basename(doc.metadata.get("source", "unknown document"))
        page = doc.metadata.get("page", "unknown page")

        ref_key = f"{source}:{page}"
        if ref_key in seen:
            continue
        seen.add(ref_key)

        text = doc.page_content.strip().replace("\n", " ")
        block = f"[{source} | page {page}] (score: {score:.3f})\n{text}\n"
        context_blocks.append(block)

    formatted_context = "\n\n".join(context_blocks)
    return formatted_context

if __name__ == "__main__":
    create_vectorstore("data/docs")
