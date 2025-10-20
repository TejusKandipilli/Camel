import time
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def create_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        add_start_index=True,
    )

def load_chunk_pdf(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    splitter = create_splitter()
    start = time.time()
    documents = loader.load_and_split(splitter)
    chunk_time = time.time() - start

    for i, doc in enumerate(documents):
        doc.metadata["source_file"] = pdf_path
        doc.metadata["chunk_index"] = i

    print(f"📄 Loaded PDF and split into {len(documents)} chunks in {chunk_time:.2f}s")
    return documents

def add_chunks_to_vc(pdf_path, collection_name, avg_chunk_time=0.1):
    """
    avg_chunk_time: seconds per chunk (adjust based on previous runs)
    """
    start_total = time.time()
    try:
        documents = load_chunk_pdf(pdf_path)
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )

        # Check for existing documents
        existing = vector_store.get(where={"source_file": pdf_path})
        if existing and len(existing["ids"]) > 0:
            print(f"⚠️ {pdf_path} already exists in '{collection_name}'. Skipping.")
            return existing["ids"]

        start_embedding = time.time()
        ids = vector_store.add_documents(documents)
        embedding_time = time.time() - start_embedding

        total_time = time.time() - start_total

        print(f"✅ Added {len(documents)} chunks from {pdf_path} to '{collection_name}'")
        print(f"⏱ Actual embedding & insertion took ~{embedding_time:.2f}s")
        print(f"⏱ Total runtime: ~{total_time:.2f}s")
        print(f"📊 Chunks per second: {len(documents)/embedding_time:.2f}")

        return ids
    except Exception as e:
        total_time = time.time() - start_total
        print(f"❌ Error adding {pdf_path}: {str(e)}")
        print(f"⏱ Total runtime before error: ~{total_time:.2f}s")
        return []
