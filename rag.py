from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import (
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
import pathlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


embedder = OllamaEmbeddings(
    model="nomic-embed-text"
)

CHUNK_SIZE = 400                    
CHUNK_OVERLAP = 100             
TOP_K = 6
SCORE_THRESHOLD = 1.5
VECTOR_STORE_PATH = "vector_store"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)


def load_documents(data_dir: str = "data/") -> list[Document]:
    """Load all supported documents from directory"""
    docs = []
    supported_extensions = {".pdf", ".txt", ".docx", ".doc"}
    
    for file_path in pathlib.Path(data_dir).glob("*.*"):
        if file_path.suffix.lower() not in supported_extensions:
            continue 
            
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            loader = UnstructuredWordDocumentLoader(str(file_path))
        
        docs.extend(loader.load())
    
    return docs



def get_vector_store():
    """Load vector store if exists, otherwise ingest documents"""
    if not hasattr(get_vector_store, "vector_store"):
        if pathlib.Path(VECTOR_STORE_PATH).exists():
            print("Loading existing vector store...")
            get_vector_store.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, embedder, allow_dangerous_deserialization=True
            )
        else:
            docs = load_documents()
            chunks = text_splitter.split_documents(docs)
            get_vector_store.vector_store = FAISS.from_documents(chunks, embedder)
            get_vector_store.vector_store.save_local(VECTOR_STORE_PATH)
            print(f"Created vector store with {len(chunks)} chunks from {len(docs)} documents.")
    
    return get_vector_store.vector_store

def ingest():
    """Public function called by Streamlit app to initialize the vector store"""
    get_vector_store()

def retrieve_docs(query: str, k: int = TOP_K):
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    
    filtered = [(doc, score) for doc, score in results if score <= SCORE_THRESHOLD]
    
    if not filtered:
        return "No relevant information found.", []
    
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in filtered)
    sources = [doc.metadata for doc, _ in filtered]
    
    return context, sources
