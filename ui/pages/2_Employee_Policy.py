__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import nltk

nltk_data_dir = "./resources/nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

import chromadb
import streamlit as st
from streamlit_feedback import streamlit_feedback


from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate

from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer

from llama_index.vector_stores.chroma import ChromaVectorStore
from io import BytesIO
# import mysql.connector
# from mysql.connector import Error

import re
import pytz
import uuid
import yaml
from datetime import datetime
from src.utils import preprocess_input

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

temperature = st.secrets['knowledge_assistant_tmp']
generation_config = {"temperature": temperature}
# safety_settings = 

gemini_api_key = st.secrets["gemini_api_key"]
text_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/text-embedding-004")
llm = Gemini(api_key=gemini_api_key, model_name='models/gemini-1.5-flash', generation_config=generation_config)

# Global settings
# Settings.llm = model
Settings.embed_model = text_embedding_model
Settings.context_window = st.secrets['knowledge_assistant_cw']

# Load YAML data
# config = yaml.safe_load(open('config/config.yaml', 'r'))

### Load Chroma
collection_name = "emp-policy-20240905"

client = chromadb.PersistentClient(path="./emp_policy_db")

chroma_collection = client.get_collection(collection_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, 
                                           storage_context=storage_context,
                                           embed_model = text_embedding_model
)


llm_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge"
    "You are Virtual Assitant of Bank XYZ. Answer in Indonesian Language"
    "Give the reference based on page label in format Answer first then Sumber: \n<list of unique and relevant 'page_label' from Reference Documents in bullets format, like Halaman + pagel_label> below the answer"
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

opening_content = """Selamat pagi/siang/sore/malam! 👋  \nApa yang ingin Anda tanyakan tentang Employee Policy?  \nSaya siap membantu Anda dengan informasi mengenai:  \n1. Apa saja hak karyawan.  \n2. Bagaimana sistem cuti.  \n3. Kebijakan Tunjangan Hari Raya."""

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: llm
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        metadata = [x.metadata for x in nodes]
        
        # Extracting the pairs
        title_url_pairs = set((item['file_name'], item['page_label']) for item in metadata)

        reference_str = "\n\nReference Documents:\n"
        for i, (title, page_label) in enumerate(title_url_pairs, 1):
            reference_str += f"{i}. Filename: {title}\n   Halaman: {page_label}\n"

        context_str = context_str + reference_str
        response = self.llm.complete(
            llm_prompt.format(context_str=context_str, query_str=query_str)
        )

        return response, nodes

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": f"{opening_content}"}]

def generate_gemini_response(prompt_input, selected_option):

    if selected_option == "gemini-1.5-pro":
        model_name = "models/gemini-1.5-pro"
    else:
        model_name = "models/gemini-1.5-flash"

    llm_model = Gemini(api_key=gemini_api_key, model_name=model_name, generation_config=generation_config)

    retriever = index.as_retriever(similarity_top_k=5, retriever_mode="rake")
    synthesizer = get_response_synthesizer(response_mode="simple_summarize", llm=llm_model)

    query_engine = RAGStringQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        llm=llm_model,
        llm_prompt=llm_prompt,
    )

    response = query_engine.query(prompt_input)

    return response[0], [x.metadata for x in response[1]]

def get_url_from_title(yaml_data, search_title):
    for category, details in yaml_data.items():
        for detail_page in details['detail_pages']:
            if detail_page['title'] == search_title:
                return detail_page['url']
    return ''
        
def unique_preserve_order(input_list):
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def get_sources(metadata):
    # Extract the list of unique titles from metadata using a set
    titles = [info['title'] for info in metadata.values()]
    unique_titles = unique_preserve_order(titles)
    # unique_titles = unique_titles[:5]
    text = ''
    for title in unique_titles:
        url = get_url_from_title(config, title)
        text += url + "  \n"

    return text

# def create_connection():
#     connection = None
#     try:
#         connection = mysql.connector.connect(
#             host=host_name,
#             user=username,
#             password=user_password,
#             database=db_name,
#             port=port
#         )
#         if connection.is_connected():
#             print("Connection to MySQL DB successful")
#     except Error as e:
#         print(f"The error '{e}' occurred")
#     return connection


# def store_feedback(question, response, response_type, user_feedback, timestamp, session_id):
#     conn = create_connection()
#     cursor = conn.cursor()
#     try:
#         cursor.execute('''INSERT INTO feedback (question, response, response_type, user_feedback, timestamp, session_id) VALUES (%s, %s, %s, %s, %s, %s)''', 
#             (question, response, response_type, user_feedback, timestamp, session_id))
#         conn.commit()
#         conn.close()
#         print("Query executed successfully")
#     except Error as e:
#         print(f"The error '{e}' occurred")
#         conn.rollback()


# Feedback mechanism using streamlit-feedback
def handle_feedback(user_response, result):
    st.toast("✔️ Feedback received!")

    response_type = 'good' if user_response['score']=='👍' else 'bad'
    feedback = user_response['text']

    jakarta_tz = pytz.timezone('Asia/Jakarta')

    timestamp = datetime.now(jakarta_tz)
    timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')

    # Store feedback in the database
    # store_feedback(st.session_state.messages[-2]["content"], result, response_type, feedback, timestamp_str, st.session_state.session_id)
          

    # Reset session ID after feedback is submitted
    st.session_state["session_id"] = str(uuid.uuid4())


def main():
    # App title
    st.set_page_config(layout="wide", page_title="Xaira Chatbot RAG🧕💬")
    st.title("Employee Policy Assistant Bank XYZ")

    # Clear chat everytime pages move
    # clear_chat_history()
    
    with st.sidebar:
        options = ["gemini-1.5-flash", "gemini-1.5-pro"]
        selected_option = st.selectbox("Select Gemini Model:", options, index= 0)
        st.write("""Aisyah bisa bantu kamu mengetahui informasi promo dan layanan berikut:  
            \n1. Info Cuti 
            \n2. Info Tunjangan Hari Raya""")
        # Main content area for displaying chat messages
        st.button('Clear Chat History', on_click=clear_chat_history)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": f"{opening_content}"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                placeholder = st.empty()

                # Error handling
                try:
                    llm_response = generate_gemini_response(preprocess_input(prompt), selected_option)
                    response = str(llm_response[0])
                    # metadata = llm_response[1]
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)

                    # sources = get_sources(metadata)
                    # full_response += "\n\n Sumber:  \n" + sources

                except:
                    full_response = full_response = "Maaf, saya tidak dapat menemukan jawaban pertanyaan anda. Silahkan parafrase atau tanyakan kembali dalam beberapa saat."
            
                placeholder.markdown(full_response) 

        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

        full_response = st.session_state.messages[-1]['content']
                    
        # Feedback submission form
        feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                key=st.session_state.session_id,
                on_submit=handle_feedback,
                align="flex-end",
                kwargs={"result": full_response},
                        )

if __name__ == '__main__':
    main()