from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Updated import path
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI

from dotenv import load_dotenv
import os
from Chatbot import run_chatbot
# Load environment variables
load_dotenv()

# Retrieve OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# === Load PDF ===
loader = PyPDFLoader("DataSRC/DeclarationofIndependence.pdf")
documents = loader.load()

# === Split documents into chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# === Create embeddings ===
embeddings = OpenAIEmbeddings()

# === Store embeddings in ChromaDB ===
# Chroma will automatically persist to the directory
# No need to call persist() in newer versions
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# === Create retriever ===
retriever = db.as_retriever(search_kwargs={"k": 3})

# === Create RAG QA chain ===
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)



# === Run chatbot ===
if __name__ == "__main__":
    run_chatbot(qa)