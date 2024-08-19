import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import tempfile
import dill as pickle

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import question_answering

# Load environment variables
load_dotenv()

# Load the Google API key from environment variables
google_api_key = os.getenv('GOOGLEAI_API_KEY')

if google_api_key is None:
    st.error('Google AI API key is not set.')
    st.stop()

def init():
    st.set_page_config(
        page_title="Your Personal AI Assistant",
        page_icon="random",
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Report a bug': "mailto:amaobiokeoma@gmail.com",
            'About': """ 
            My name is Okey Amy, an ML Engineer passionate about AI and all things tech.
            I love building tools that make life easier and smarter. With a background in machine 
            learning and experience in creating interactive AI assistants, I'm excited to share my latest 
            project—a personal AI assistant that helps with general knowledge and document analysis.

"""
        }
    )

def load_document(document_file):
    """
    Load the uploaded document and create a FAISS vector store.
    """
    if document_file is not None:
        # Determine the file type and use appropriate loader
        file_type = document_file.name.split('.')[-1].lower()
        if file_type == 'txt':
            loader = TextLoader
        elif file_type == 'pdf':
            loader = PyPDFLoader
        elif file_type == 'docx':
            loader = Docx2txtLoader
        else:
            st.error('Unsupported file format.')
            return None

        # Save the uploaded document to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            temp_file.write(document_file.read())
            tmp_file_path = temp_file.name

        # Load the document
        loader_instance = loader(tmp_file_path)
        document = loader_instance.load_and_split()

        # Generate embeddings and create FAISS vector store
        embeddings = GooglePalmEmbeddings(show_progress_bar=True, google_api_key=google_api_key)
        vectorstore = FAISS.from_documents(documents=document, embedding=embeddings)

        # Save the vector store to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            vectorstore_path = temp_file.name
            with open(vectorstore_path, 'wb') as f:
                pickle.dump(vectorstore, f)

        return vectorstore_path

def main():
    init()
    
    st.header('Your Personal AI Assistant 🤖')

    # Initialize the chat model
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1, api_key=google_api_key)

    # Initialize the session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # System instruction
    system_instruction = SystemMessage(content=''' 
                                       You are a resourceful AI that acts like a human who is going to assists users with their 
                                       queries by providing accurate information. 
                                       If you don't have the information directly, you will automatically search for and 
                                       provide a relevant website link that may contain the answer, without stating your limitations. 
                                       You can not use however,moreover, and other in your response. Always remind the user to refresh the 
                                       page if they want to upload a new document.
                                       
    ''')

    # Add system instruction to the beginning of the message history if not present
    if not st.session_state.messages or not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(0, system_instruction)

    # User input
    user_input = st.chat_input("Type your message here...")

    # User uploads document
    with st.sidebar.header('Upload your document'):
        upload_document = st.file_uploader('Upload your document here')

    # Allow external information (toggle button in sidebar)
    st.sidebar.header('Settings')
    allow_external = st.sidebar.checkbox("Allow external information", value=False)

    # Load the document and create a vector store if not already loaded
    if upload_document and 'vectorstore_path' not in st.session_state:
        with st.spinner("Loading your document..."):
            st.session_state.vectorstore_path = load_document(upload_document)
            st.session_state.retriever = None

    # Load the retriever from the saved vector store
    if 'vectorstore_path' in st.session_state and st.session_state.retriever is None:
        with open(st.session_state.vectorstore_path, 'rb') as f:
            vectorstore = pickle.load(f)
            st.session_state.retriever = vectorstore.as_retriever()

    # Process user input and provide responses
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))

        if 'retriever' in st.session_state and not allow_external:
            retriever = st.session_state.retriever
            qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=chat,
                chain_type="stuff",
                retriever=retriever
            )
            try:
                matching_results = retriever.get_relevant_documents(user_input)
                chain = question_answering.load_qa_chain(chat, chain_type='stuff')
                response = chain.run(input_documents=matching_results, question=user_input)
            except Exception as e:
                response = chat(st.session_state.messages).content
        else:
            # Filter out SystemMessage from the chat history
            filtered_messages = [msg for msg in st.session_state.messages if not isinstance(msg, SystemMessage)]
            with st.spinner('Thinking...'):
                response = chat(filtered_messages).content

        with st.spinner('Thinking...'):
            st.session_state.messages.append(AIMessage(content=response))

    # Display the chat history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=str(i) + '_user')
        elif isinstance(msg, AIMessage):
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == "__main__":
    main()
