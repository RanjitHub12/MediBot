import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

# --- LangChain Imports ---
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- Load Environment Variables ---
# Make sure your .env file with GROQ_API_KEY is in the same directory
load_dotenv(find_dotenv())

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_NAME = "llama-3.1-8b-instant"

# --- Caching the QA Chain ---
# This is a key Streamlit concept. It ensures that we load the model and databasels
# only once, making the app much faster on subsequent interactions.
@st.cache_resource
def load_qa_chain():
    """
    Loads the QA chain with the Groq LLM and the FAISS vector store.
    The @_st.cache_resource decorator ensures this function only runs once.
    """
    print("Loading QA chain... This will happen only once.")
    
    # Load the vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    
    # Initialize the Groq LLM
    llm = ChatGroq(model=MODEL_NAME, temperature=0.5, max_tokens=512)
    
    # Create a custom prompt
    prompt_template = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# --- Streamlit App UI ---

# Set the page title and icon
st.set_page_config(page_title="Chat with Your Docs", page_icon="📄")

# Display the title
st.title("📄 Chat with Your Documents")
st.write("Ask questions about the content of your PDFs, and the AI will find the answers.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load the QA chain (this will be cached)
qa_chain = load_qa_chain()

# Get user input from the chat input box
if prompt := st.chat_input("Ask a question..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant's response
    with st.chat_message("assistant"):
        # Show a thinking spinner while processing
        with st.spinner("Thinking..."):
            # Get the response from the QA chain
            response = qa_chain.invoke({'query': prompt})
            
            # Display the main result
            st.markdown(response["result"])
            
            # Display the source documents in an expander
            with st.expander("View Sources"):
                st.write("The following sources were used to generate the answer:")
                for doc in response["source_documents"]:
                    # Clean up the page content for display
                    page_content = doc.page_content.replace('\n', ' ').strip()
                    st.info(f"**Page {doc.metadata.get('page', 'N/A')}:** \"{page_content[:250]}...\"")

    # Add the assistant's full response to session state
    assistant_response = {
        "role": "assistant",
        "content": response["result"] # We only store the main answer in history
    }
    st.session_state.messages.append(assistant_response)