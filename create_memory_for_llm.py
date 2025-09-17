# from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # Step 1: loading raw PDF
# DATA_PATH="data/"
# def load_pdf_files(data):
#     loader= DirectoryLoader(data, glob='*.pdf', loader_cls=PyMuPDFLoader)
#     documents=loader.load()
#     return documents

# documents=load_pdf_files(data=DATA_PATH)
# # print("Length of PDF pages: ", len(documents))

# # Step 2: creating chunks of these data
# def create_chunks(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks

# text_chunks=create_chunks(extracted_data=documents)
# # print("Length of Text Chunks: ", len(text_chunks))

# # Step 3: vector embedding
# def get_embedding_model():
#     embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return embedding_model

# embedding_model=get_embedding_model()

# # Step 4: storing embeddings in FAISS
# DB_FAISS_PATH="vectorstore/db_faiss"
# db=FAISS.from_documents(text_chunks, embedding_model)
# db.save_local(DB_FAISS_PATH)

import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define constants for paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    """
    This function orchestrates the entire process of creating the vector database.
    It loads PDFs, splits them into chunks, creates embeddings, and saves the result.
    """
    # Step 1: Check for data directory and load PDF files
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: The directory '{DATA_PATH}' was not found.")
        print("Please create it and add your PDF files.")
        return

    print(f"🔄 Loading documents from '{DATA_PATH}'...")
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyMuPDFLoader)
    documents = loader.load()

    if not documents:
        print(f"⚠️ No PDF documents found in '{DATA_PATH}'. The process will stop.")
        return
    
    print(f"✅ Loaded {len(documents)} document pages.")

    # Step 2: Create chunks of the data
    print("🔄 Splitting documents into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(text_chunks)} text chunks.")

    # Step 3: Get the embedding model
    # This model runs locally on your machine for free.
    print("🔄 Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model initialized.")

    # Step 4: Create embeddings and store them in FAISS
    print("🔄 Creating embeddings and building the FAISS vector store. This may take a moment...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"🎉 Vector store successfully created and saved at '{DB_FAISS_PATH}'.")


# This is the standard entry point for a Python script.
if __name__ == "__main__":
    create_vector_db()