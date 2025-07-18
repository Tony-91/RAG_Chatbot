# RAG Chatbot for Document Analysis

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the Declaration of Independence by analyzing its content. The application uses LangChain, OpenAI embeddings, and ChromaDB for vector storage.

## Features

- PDF document ingestion and processing
- Text chunking with overlap for context preservation
- Vector embeddings using OpenAI's API
- Local vector storage with ChromaDB
- Interactive command-line chat interface
- Source citation for all answers

## Prerequisites

- Python 3.8+
- OpenAI API key
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG_Chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install them manually:
   ```bash
   pip install langchain langchain-community langchain-openai chromadb pypdf python-dotenv
   ```

4. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Project Structure

```
RAG_Chatbot/
├── .env                    # Environment variables
├── .gitignore             # Git ignore file
├── main.py                # Main application script
├── Chatbot.py             # Chat interface implementation
├── DataSRC/               # Directory for source documents
│   └── DeclarationofIndependence.pdf
└── chroma_db/             # Vector store data (created at runtime)
```

## How It Works

### 1. Document Ingestion

```python
# Load PDF
loader = PyPDFLoader("DataSRC/DeclarationofIndependence.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)
```

### 2. Embeddings & Vector Storage

```python
# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create and store vectors
db = Chroma.from_documents(
    docs, 
    embeddings, 
    persist_directory="./chroma_db"
)
```

### 3. RAG Chain Creation

```python
# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = OpenAI(temperature=0)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

### 4. Chat Interface

```python
def run_chatbot(qa_chain):
    print("📜 Declaration of Independence RAG Chatbot ready. Ask anything about the PDF (type 'exit' to quit).")
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = result["source_documents"]
        
        print(f"\n🤖 Answer: {answer}\n")
        print("📄 Sources:")
        for src in sources:
            print(f"- {src.metadata.get('source', 'Unknown Source')}")
        print("\n" + "="*50 + "\n")
```

## Running the Application

1. Ensure your virtual environment is activated
2. Run the main script:
   ```bash
   python main.py
   ```
3. Type your questions about the Declaration of Independence
4. Type 'exit' to quit

## Example Usage

```
📜 Declaration of Independence RAG Chatbot ready. Ask anything about the PDF (type 'exit' to quit).

You: Who wrote the Declaration of Independence?

🤖 Answer: The Declaration of Independence was primarily written by Thomas Jefferson, with input from John Adams, Benjamin Franklin, Roger Sherman, and Robert Livingston.

📄 Source: DataSRC/DeclarationofIndependence.pdf

==================================================
```

## Customization

- To use a different PDF, replace the file in the `DataSRC` directory and update the path in `main.py`
- Adjust `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` for different document processing needs
- Modify `temperature` in the OpenAI initialization to control response creativity (0 = most deterministic)

## Dependencies

- langchain
- langchain-community
- langchain-openai
- chromadb
- pypdf
- python-dotenv

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Uses [OpenAI's API](https://platform.openai.com/) for embeddings and LLM
- Vector storage powered by [ChromaDB](https://www.trychroma.com/)
