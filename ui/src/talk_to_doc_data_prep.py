from src.document_reader import DOCUMENTReader
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings, StorageContext
import faiss

gemini_api_key = st.secrets["gemini_api_key"]
text_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/text-embedding-004")
Settings.embed_model = text_embedding_model
d = 768
faiss_index = faiss.IndexFlatL2(d)

class Prep:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001",
        #     google_api_key=gemini_api_key,
        # )

    def ingest_documents(
        self,
        file: str,
    ):
        
        loader = DOCUMENTReader()
        chunks,chunks_faiss = loader.load_document(file_path=file)
        
        # Initialize the vector store
        # faiss_vector_store = FaissVectorStore.from_documents(chunks_faiss, emb="models/text-embedding-004",google_api_key=gemini_api_key)

        faiss_vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=faiss_vector_store)
        index = VectorStoreIndex.from_documents(chunks_faiss, storage_context=storage_context, embed_model = text_embedding_model)
        
        vector_store = VectorStoreIndex.from_documents(chunks)
        return vector_store, faiss_vector_store