import os
import streamlit as st
# from dotenv import load_dotenv
# from langchain.document_loaders.pdf import PyPDFLoader
from copy import deepcopy
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
from llama_index.core import Document as dc

# document_charsplitter_chunksize = st.secrets["DOCUMENT_CHARSPLITTER_CHUNKSIZE"]
# document_charsplitter_chunk_overlap = st.secrets["DOCUMENT_CHARSPLITTER_CHUNK_OVERLAP"]
llama_parse_key = st.secrets["LLAMA_PARSE_KEY"]

# load_dotenv()
class DOCUMENTReader:
    """Custom PDF Loader to embed metadata with the pdfs."""

    def __init__(self) -> None:
        self.file_name = ""
        self.total_pages = 0

    def get_page_nodes(self,docs, separator="\n---\n"):
        """Split each document into page node, by separator."""
        nodes = []
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                node = TextNode(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                nodes.append(node)

        return nodes
    
    def split_and_chunk_document(self,documents):
        page_nodes = self.get_page_nodes(documents)

        chunks = []
        chunks_faiss = []
        no_chunk = 0
        no_faiss = 0
        for i,text in enumerate(page_nodes):
            if text.get_text() == "":
                no_chunk += 1
            else :
                no_chunk += 1
                chunks.append(
                dc(
                    text=text.get_text(),
                    metadata=
                            {"file_name": self.file_name,
                            "page_no": str(no_chunk)}))
        
        for i,text in enumerate(page_nodes):
            if text.get_text() == "":
                no_faiss += 1
            else :
                no_faiss += 1
                chunks_faiss.append(
                dc(
                    page_content=text.get_text(),
                    metadata=
                            {"file_name": self.file_name,
                            "page_no": str(no_faiss)}))

            # text_by_loaders_page.append(text_temp)
        return chunks,chunks_faiss

    def load_document(self, file_path):
        # Get the filename from file path
        self.file_name = os.path.basename(file_path)

        parser = LlamaParse(
        api_key= llama_parse_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,  # Optionally you can define a language, default=en
        )


        loader = parser.load_data(file_path)


        final_chunks,chunks_faiss = self.split_and_chunk_document(loader)
        return final_chunks,chunks_faiss
