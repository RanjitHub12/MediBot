# import os

# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())


# # Step 1: Setup LLM (Mistral with HuggingFace)
# HF_TOKEN=os.environ.get("HF_TOKEN")
# # HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
# HUGGINGFACE_REPO_ID="google/gemma-7b-it"

# # def load_llm(huggingface_repo_id):
# #     llm=HuggingFaceEndpoint(
# #         repo_id=huggingface_repo_id,
# #         temperature=0.5,
# #         model_kwargs={"token":HF_TOKEN,
# #                       "max_length":"512"}
# #     )
# #     return llm
# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         huggingfacehub_api_token=HF_TOKEN,  # Correct parameter for authentication
#         temperature=0.5,
#         max_new_tokens=512                  # Correct parameter for length
#     )
#     return llm
# # Step 2: Connect LLM with FAISS and Create chain

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context

# Context: {context}
# Question: {question}

# Start the answer directly. No small talk please.
# """

# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Load Database
# DB_FAISS_PATH="vectorstore/db_faiss"
# embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Create QA chain
# qa_chain=RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k':3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Now invoke with a single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])


import os
from dotenv import load_dotenv, find_dotenv

# --- LangChain Imports ---
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- NEW: Import the Groq LLM ---
from langchain_groq import ChatGroq

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
# MODEL_NAME = "gemma-7b-it"  # We can still use Gemma through the Groq API
MODEL_NAME = "llama-3.1-8b-instant"

# --- Step 1: Setup LLM (Now using Groq) ---
def load_llm(model_name):
    """
    Loads the Language Model from Groq.
    Make sure your .env file has GROQ_API_KEY.
    """
    llm = ChatGroq(
        model=model_name,
        temperature=0.5,
        max_tokens=512
        # The API key is automatically read from the GROQ_API_KEY environment variable
    )
    return llm

# --- Step 2: Define the Prompt Template ---
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt():
    """Creates the PromptTemplate from the custom template string."""
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    return prompt

# --- Main Execution ---
if __name__ == "__main__":
    # Load the locally stored vector database
    print("🔄 Loading vector database...")
    # The embedding model must be the same one used to create the database
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("✅ Database loaded successfully.")

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(MODEL_NAME),  # This now correctly loads the Groq LLM
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )

    # Start the conversation loop
    print("\n--- RAG Q&A is Ready ---")
    while True:
        user_query = input("\n📝 Write Query Here (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        # Get the response from the chain
        response = qa_chain.invoke({'query': user_query})
        
        # Print the results
        print("\n💡 RESULT:")
        print(response["result"])
        
        print("\n📚 SOURCE DOCUMENTS:")
        for doc in response["source_documents"]:
            # Tries to get the page number from metadata for better source tracking
            page = doc.metadata.get('page', 'N/A')
            print(f"- Page {page}: {doc.page_content[:150]}...")