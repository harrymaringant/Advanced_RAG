import streamlit as st
import os
from src.talk_to_doc_data_prep import Prep
from src.qachain import QAChain
from streamlit_feedback import streamlit_feedback
import tempfile
from pathlib import Path
import shutil
import nest_asyncio
from datetime import datetime
import pytz
import uuid


nest_asyncio.apply()
# extracting text from document

# @st.cache_resource(show_spinner=False)
def get_document_text(doc):
    ingestion = Prep()
    vector_store,f_store = ingestion.ingest_documents(doc)
    return vector_store,f_store

def get_conversationchain(query,selected_option,vector, faiss_vector):
    if selected_option == "gemini-1.5-pro":
        model = "models/gemini-1.5-pro"
    else:
        model = "models/gemini-1.5-flash"

    qna = QAChain(model,vector,faiss_vector)
    results = qna.generate_response(
        query=query
    )
    return results

def clear_vector_db():
    st.session_state.messages = [{"role": "assistant", "content": "upload some documents and ask me a question"}]
    abs_path = os.path.dirname(os.path.abspath(__file__))
    CurrentFolder = str(Path(abs_path).resolve())
    path = os.path.join(CurrentFolder, "database")
    shutil.rmtree(path)

# generating response from user queries and displaying them accordingly
# def handle_question(question):
#     response=st.session_state.conversation({'question': question})
#     st.session_state.chat_history=response["chat_history"]
#     for i,msg in enumerate(st.session_state.chat_history):
#         if i%2==0:
#             st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some documents and ask me a question"}]

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
    st.set_page_config(page_title="Chat with multiple DOCUMENTs",page_icon="🤖")
    st.title("AIDSU - Chat With Documents")

    # Clear chat everytime pages move
    # clear_chat_history()

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "f_store" not in st.session_state:
        st.session_state.f_store = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Upload your Documents files to start."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    with st.sidebar:

        options = ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
        selected_option = st.selectbox("Select Gemini Model:", options, index= 1)

        docs = st.file_uploader("File upload", type= ['pdf', 'docx'] ,accept_multiple_files=True)
        # print(selected_option)
        if st.button("Process"):
                if docs:
                    for doc in docs:
                        temp_dir = tempfile.mkdtemp()
                        path = os.path.join(temp_dir, doc.name)
                        print(path)
                        with open(path, "wb") as f:
                            f.write(doc.getvalue())
                        #extract from document -> get the text chunk -> create vectore store
                        st.session_state.vector_store = get_document_text(path)
                        st.session_state.messages.append({"role": "assistant", "content": "Your PDFs have been processed. Ask your questions now!"})
                    st.success("Done")
                else:
                    st.warning("Please upload PDF files first.")
        
    # Main content area for displaying chat messages
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    # st.sidebar.button('Clear VectorDB', on_click=clear_vector_db)
    
    user_question = st.chat_input("Ask a question about the PDF...")

    if user_question and st.session_state.vector_store:
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        response = get_conversationchain(user_question,selected_option,st.session_state["vector_store"])

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

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



    # if "messages" not in st.session_state.keys():
    #     st.session_state.messages = [
    #         {"role": "assistant", "content": "upload some documents and ask me a question"}]


    # if prompt := st.chat_input():
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.write(prompt)

    #     # Display chat messages and bot response
    # if st.session_state.messages[-1]["role"] != "assistant":
    #     with st.chat_message("assistant"):
    #         with st.spinner("Thinking..."):
    #             response = get_conversationchain(prompt,selected_option)
    #             placeholder = st.empty()
    #             full_response = ''
    #             for item in list(response):
    #                 full_response += item
    #                 placeholder.markdown(full_response)
    #             placeholder.markdown(full_response)
    #     if response is not None:
    #         message = {"role": "assistant", "content": full_response}
    #         st.session_state.messages.append(message)

if __name__ == '__main__':
    main()