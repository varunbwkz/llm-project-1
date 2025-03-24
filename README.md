# Enhanced RAG System

A powerful Retrieval-Augmented Generation (RAG) system that allows you to upload PDF documents and query them using natural language. This application combines the power of Large Language Models (LLMs) with a vector database to provide accurate answers based solely on your documents.

## Features

- **PDF Document Processing**: Upload and process multiple PDF documents
- **Smart Text Chunking**: Automatically divides documents into manageable chunks with configurable size and overlap
- **Vector Search**: Uses semantic similarity to find the most relevant document sections
- **AI-Powered Answering**: Generates comprehensive answers with direct responses, detailed explanations, and key points
- **Source Tracking**: All answers include references to source documents
- **User-Friendly Interface**: Clean, intuitive Streamlit interface with helpful tooltips
- **Document Management**: Add or remove documents as needed
- **Query History**: Review previous questions and answers
- **Advanced Settings**: Customize chunk size, overlap, and temperature for optimal results
- **API Access**: Query the system programmatically via REST API endpoints

## 🔧 Technologies Used

- **Frontend**: Streamlit
- **API Server**: FastAPI
- **Vector Database**: ChromaDB
- **Embedding Model**: OpenAI's text-embedding-3-small
- **LLM**: OpenAI's GPT-4o-mini
- **PDF Processing**: PyPDF2
- **Parallel Processing**: Python's concurrent.futures

## 📋 Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. **Clone the repository**:
   ```bash
   git clone (this repo)
   cd enhanced-rag-system (if you want to keep it in your own folder)
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   
   For Streamlit interface:
   ```bash
   pip install -r requirements.txt
   ```
   
   For API server:
   ```bash
   pip install -r requirements_api.txt
   ```

4. **Create a `.env` file** in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   API_PORT=8000  # Optional: customize the API port (default: 8000)
   ```

## 🏃‍♂️ Usage

### Streamlit Web Interface

1. **Start the application**:
   ```bash
   streamlit run rag_query_system.py
   ```

2. **Upload documents**:
   - Navigate to the "Document Upload" tab
   - Click "Browse files" and select one or more PDF files
   - Wait for processing to complete

3. **Query your documents**:
   - Switch to the "Query Documents" tab
   - Enter your question in the text area
   - Adjust options if needed:
     - **Passages to retrieve**: Number of document chunks to consider (higher = more context but potentially less focused)
     - **Temperature**: Controls randomness in the AI's response (lower = more deterministic, higher = more creative)
   - Click "Submit Query"

4. **Review answers**:
   - The response is divided into three sections:
     - **Direct Answer**: A concise response to your question
     - **Detailed Explanation**: In-depth analysis with supporting evidence
     - **Key Points**: Important takeaways from the documents
   - Click "View Source Passages" to see the exact document sections used to generate the answer

### API Server

1. **Start the API server**:
   ```bash
   python api_server.py
   ```
   The server will run on http://localhost:8000 by default.

2. **View API documentation**:
   Open http://localhost:8000/docs in your browser to see the interactive API documentation.

3. **API Endpoints**:

   - `GET /` - Check if the API is running
   - `GET /documents` - List all documents in the system
   - `POST /query` - Query the documents with natural language
   - `GET /info` - Get information about the models being used

4. **Example API Request**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/query' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
     "query": "What is the main topic of the document?",
     "n_results": 3,
     "temperature": 0.7
   }'
   ```

## ⚙️ Configuration Options

### Document Chunking

In the "Advanced Options" section of the Document Upload tab, you can adjust:

- **Chunk Size** (500-2000 characters): Think of this like breaking a big book into smaller sections. A larger chunk size means bigger sections (more context but might be less precise), while smaller chunks mean shorter sections (more precise but might miss context). Example: with 1000, a page might be split into 3-4 parts.

- **Chunk Overlap** (0-500 characters): This is like having each section share a few sentences with the next section, so we don't lose the connection between ideas. Example: if overlap is 200, each chunk will share about 200 characters with the next chunk to maintain context.

### Response Generation

When querying documents, you can adjust:

- **Passages to retrieve** (1-10): The number of relevant document passages to use
- **Temperature** (0.0-1.0): Controls the randomness/creativity of the AI
    Think of temperature like a creativity dial. At 0.0 (cold), the AI gives consistent, focused answers - great for factual queries. At 1.0 (hot), it gets more creative and varied - better for brainstorming. Example: For 'What is quantum computing?', 0.2 gives technical definitions, while 0.8 might include analogies and examples.

## 🧠 How It Works

This application implements a RAG (Retrieval-Augmented Generation) architecture:

1. **Document Processing**:
   - PDFs are converted to text
   - Text is divided into smaller chunks with configurable overlap
   - Each chunk is assigned a unique ID and source reference

2. **Vector Embedding**:
   - Document chunks are converted to vector embeddings using OpenAI's embedding model
   - These vectors capture the semantic meaning of the text
   - Embeddings are stored in a ChromaDB vector database

3. **Query Processing**:
   - User questions are converted to the same vector space
   - Vector similarity search finds the most relevant document chunks
   - Retrieved chunks are organized with source information

4. **Response Generation**:
   - The LLM receives the question and relevant document chunks
   - A carefully crafted prompt ensures the model only uses information from the documents
   - The response is structured into direct answer, detailed explanation, and key points sections

## 📁 Project Structure

```
enhanced-rag-system/
├── rag_query_system.py        # Main application file (Streamlit UI)
├── api_server.py              # API server (FastAPI)
├── requirements.txt           # Dependencies for Streamlit UI
├── requirements_api.txt       # Dependencies for API server
├── .env                       # Environment variables (API keys)
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
└── chroma_db/                 # Vector database storage (generated on its own)
```

## 🔍 Classes and Components

- **TheModelSelector**: Manages AI model configurations
- **ThePDFProcessor**: Handles PDF reading and text chunking
- **TheRAGSystem**: Main RAG implementation with document storage, searching, and response generation

## ⚠️ Troubleshooting

### Common Issues

1. **OpenAI API Key Errors**:
   - Ensure your API key is correctly set in the `.env` file
   - Check that you have sufficient credits on your OpenAI account

2. **PDF Processing Errors**:
   - Some PDFs may have security settings that prevent text extraction
   - Try converting problematic PDFs to a different format and re-uploading

3. **Memory Issues with Large Documents**:
   - The system uses batch processing, but very large PDFs may cause memory issues.. **To be fair, i have tested with 1 document containing > 3700 pages and it still worked fine !!**
   - Consider splitting very large documents into smaller files

4. **ChromaDB Errors**:
   - If the database becomes corrupted, delete the `chroma_db` directory and restart

5. **API Server Issues**:
   - Ensure the required dependencies are installed
   - Check that the port is not already in use
   - Look for error messages in the terminal where the server is running

## Using the API in Client Applications

To integrate with the RAG System API in your applications:

### JavaScript Example

```javascript
async function queryDocuments(question) {
  const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: question,
      n_results: 3,
      temperature: 0.7
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Error: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
queryDocuments("What is the main topic of the document?")
  .then(data => {
    console.log("Direct Answer:", data.direct_answer);
    console.log("Detailed Explanation:", data.detailed_explanation);
    console.log("Key Points:", data.key_points);
    console.log("Sources:", data.sources);
  })
  .catch(error => console.error("Error:", error));
```

### Python Example

```python
import requests

def query_documents(question, n_results=3, temperature=0.7):
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "query": question,
            "n_results": n_results,
            "temperature": temperature
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")
        
    return response.json()

# Usage
try:
    result = query_documents("What is the main topic of the document?")
    print("Direct Answer:", result["direct_answer"])
    print("Detailed Explanation:", result["detailed_explanation"])
    print("Key Points:", result["key_points"])
    print("Sources:", result["sources"])
except Exception as e:
    print("Error:", str(e))
```

## 🛠️ Advanced Customization

For developers looking to extend or modify the system:

1. **Adding New Models**:
   - Extend the `TheModelSelector` class to support additional embedding or LLM models

2. **Custom Chunking Strategies**:
   - Modify the `create_chunks` method in `ThePDFProcessor` to implement different chunking algorithms

3. **Prompt Engineering**:
   - The prompt template in the `generate_response` method can be adjusted to change response style or format

4. **Extending the API**:
   - Add new endpoints to the `api_server.py` file to expose additional functionality
   - Customize response formats or add authentication as needed

## 🔮 Future Enhancements

Potential improvements for future versions:

- Support for more document formats (DOCX, TXT, HTML, etc.)
- Integration with additional LLM providers
- Local embedding model options for improved privacy
- Custom knowledge bases and persistent user profiles
- Improved document preprocessing and cleaning
- Advanced RAG techniques like reranking and query expansion
- Authentication and rate limiting for the API
- Batch document upload via API

---

For any queries, please reach out to me directly. Cheers!