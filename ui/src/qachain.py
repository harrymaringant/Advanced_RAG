# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
import streamlit as st
# from langchain.callbacks import StdOutCallbackHandler
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain, LLMChain
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.embeddings.gemini import GeminiEmbedding
# from langchain.vectorstores.deeplake import DeepLake
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings, 
#     ChatGoogleGenerativeAI
# )
gemini_api_key = st.secrets["gemini_api_key"]
llm_temperature = st.secrets["LLM_TEMPERATURE"]
top_p = st.secrets['TOP_P']
top_k = st.secrets['TOP_K']

text_embedding_model = GeminiEmbedding(api_key=gemini_api_key,model_name="models/text-embedding-004")
Settings.embed_model = text_embedding_model

# load_dotenv()
class QAChain:
    def __init__(self, model_usage,vector,faiss_vector) -> None:
        # Initialize Gemini Embeddings
        # Initialize Gemini Chat model
        self.model = Gemini(api_key=gemini_api_key, model_name=str(model_usage),generation_config={"top_k":int(top_k), "top_p":float(top_p),"temperature":float(llm_temperature)})
        
        self.vector = vector
        # Initialize GPT Cache
        # self.cache = CustomGPTCache()
        self.faiss_vector = faiss_vector
        # self.text_retriever = None

    # def ask_question(self, query):
    #     try:
    #         # Search for similar query response in cache
    #         cached_response = self.cache.find_similar_query_response(
    #             query=query, threshold=int(cache_threshold)
    #         )

    #         # If similar query response is present,vreturn it
    #         if len(cached_response) > 0:
    #             print("Using cache")
    #             result = cached_response[0]["response"]
    #         # Else generate response for the query
    #         else:
    #             print("Generating response")
    #             result = self.generate_response(query=query)
    #     except Exception as _:
    #         print("Exception raised. Generating response.")
    #         result = self.generate_response(query=query)

    #     return result

    def generate_response(self, query: str):
        # Initialize the vectorstore and retriever object
        vstore = self.vector
        f_store = self.faiss_vector

        # Write prompt to guide the LLM to generate response
        prompt_template = """You are helpful assistant of Bank XYZ, answer in 'Bahasa indonesia',
                             Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
                             provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
                             Context:\n {context_str}\n
                             Question:\n {query_str}\n

                             Answer:
                            """
        

        llm_prompt = PromptTemplate(prompt_template)
        query_engine = vstore.as_query_engine(text_qa_template=llm_prompt, llm=self.model, response_mode="simple_summarize",similarity_top_k=5)
        response = query_engine.query(query)

        answer = response.response
        # searchDocs = f_store.search(answer)
        answer_embedding = text_embedding_model.get_text_embedding(answer)
        searchDocs = f_store.query(answer_embedding)
        metadata = [j.metadata for j in searchDocs][:3]    
        page_no = ",".join(set([i['page_no']for i in metadata]))
        answer_with_source =  f"""{answer}\n\n
Sumber File : {metadata[0]['file_name']} \n\nHalaman : {page_no}"""
        # print("res : ",response.response)
        # print("met : ",response.metadata )
        return answer_with_source