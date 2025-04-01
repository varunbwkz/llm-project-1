# 📚 SmartBot Document Q&A System

This project implements an advanced Retrieval-Augmented Generation (RAG) system built with Python and Streamlit. It allows users to upload PDF documents, processes them into a searchable knowledge base, and answers questions based *only* on the information contained within those documents, providing citations and context.

## ✨ Features

*   **📄 Multi-Format Document Upload:** Accepts **PDF (`.pdf`)**, **Word (`.docx`)**, and **Text (`.txt`)** files for processing.
*   **🧠 Smart Chunking:** Splits documents into manageable, overlapping chunks, attempting to respect sentence boundaries.
*   **🚀 Parallel Processing:** Utilizes multi-threading for faster chunking of large documents.
*   **🚨 Content Safety Check:** Performs basic checks (regex patterns, OpenAI Moderation API) to flag potentially sensitive or harmful content during upload (currently warn-only).
*   **💾 Vector Storage:** Uses ChromaDB for efficient storage and retrieval of document chunks based on semantic similarity.
*   **🔍 Hybrid Search:** Combines semantic vector search (via ChromaDB) with keyword search (BM25) for potentially more robust and reliable retrieval.
*   **🤔 Query Decomposition:** Analyzes complex user questions and attempts to break them down into simpler sub-queries for more focused retrieval using an LLM.
*   **📊 Relevance Reranking:** Employs an LLM-based cross-encoder simulation to rerank retrieved document chunks for better relevance to the original query.
*   **💬 Structured Answer Generation:** Uses an OpenAI LLM (e.g., GPT-4o) to generate answers based strictly on retrieved context, formatted with clear sections (Direct Answer, Detailed Explanation, Key Points) and source citations.
*   **🚫 Hallucination Check:** Includes an optional step where a separate LLM call verifies if the generated answer is factually grounded in the provided source documents.
*   ** Citable Sources:** Displays the specific text passages used by the LLM to generate the answer.
*   **⚙️ Document Management:** Allows viewing indexed documents and deleting specific documents (and their associated chunks) from the knowledge base.
*   **🖥️ Interactive UI:** Built with Streamlit, providing a user-friendly interface for uploading, querying, and managing documents.
*   **🌡️ Configurable:** Allows adjusting LLM temperature and choosing specific documents for querying context.


## 🛠️ Technology Stack

This project utilizes the following technologies and libraries:

*   **Python:**
    *   **Purpose:** Core programming language.
    *   **Usage in this App:** Used for all backend logic, class definitions (`TheRAGSystem`, `TheDocProcessor`, etc.), and overall application scripting.

*   **Streamlit:**
    *   **Purpose:** Python library for creating interactive web applications.
    *   **Usage in this App:** Builds the entire user interface (tabs, buttons, sliders, text areas, chat display, file uploader). Manages application state (`st.session_state`) and user interaction within the `main()` function.

*   **OpenAI API:**
    *   **Purpose:** Provides access to powerful Large Language Models (LLMs) and Embedding models.
    *   **Usage in this App:** Used for generating answers (`gpt-4o` in `TheRAGSystem.generate_response`), creating text embeddings (`text-embedding-3-small` via `embedding_functions`), query decomposition (`TheQueryPlanner`), LLM reranking (`TheDocumentReranker`), hallucination checks (`TheRAGSystem._check_hallucination`), and content moderation (`moderations` endpoint in `ContentSafetyChecker`).

*   **`openai` (library):**
    *   **Purpose:** Official Python client library for interacting with the OpenAI API.
    *   **Usage in this App:** Instantiated in `TheRAGSystem`, `QueryPlanner`, and `ContentSafetyChecker` to make the actual API calls to OpenAI for embeddings, chat completions, and moderation.

*   **ChromaDB:**
    *   **Purpose:** Open-source vector database for storing and querying embeddings.
    *   **Usage in this App:** Stores document chunks and their vector embeddings persistently in the `./chroma_db` directory. Enables efficient semantic similarity search within `TheRAGSystem`.

*   **`chromadb-client` (library):**
    *   **Purpose:** Python client library for ChromaDB.
    *   **Usage in this App:** Used within `TheRAGSystem` to interact with the ChromaDB instance (e.g., `PersistentClient`, `get_or_create_collection`, `add`, `query`, `get`, `delete`).

*   **`embedding_functions` (from `chromadb.utils`):**
    *   **Purpose:** Utility from ChromaDB, often used with specific provider integrations.
    *   **Usage in this App:** The `OpenAIEmbeddingFunction` is initialized in `TheRAGSystem` and passed to the ChromaDB collection to automatically generate embeddings via the OpenAI API during document addition and querying.

*   **PyPDF2:**
    *   **Purpose:** Python library for working with PDF files.
    *   **Usage in this App:** Reads text content page by page from uploaded PDF files within the `TheDocProcessor.read_pdf` method.

*   **NLTK:**
    *   **Purpose:** Natural Language Toolkit; provides tools for text processing.
    *   **Usage in this App:** Used specifically for its sentence tokenizer (`sent_tokenize`), primarily to aid the chunking process in `TheDocProcessor` to attempt splitting text along sentence boundaries. Requires the 'punkt' resource.

*   **`rank-bm25`:**
    *   **Purpose:** Python library implementing the BM25 ranking algorithm for keyword search relevance.
    *   **Usage in this App:** Implements BM25 keyword search functionality within `TheRAGSystem._keyword_search`. This method is called as part of the `_hybrid_search` strategy to combine keyword relevance with semantic relevance.

*   **`python-dotenv`:**
    *   **Purpose:** Reads key-value pairs from a `.env` file and sets them as environment variables.
    *   **Usage in this App:** Loads the `OPENAI_API_KEY` from the `.env` file at the start of the script (`load_dotenv()`) for secure API key management.

*   **`uuid`:**
    *   **Purpose:** Python standard library for generating universally unique identifiers.
    *   **Usage in this App:** Generates unique IDs (`uuid.uuid4()`) for each document chunk created in `TheDocProcessor._process_pdf_chunk_internal` before they are added to ChromaDB.

*   **`concurrent.futures`:**
    *   **Purpose:** Python standard library for managing asynchronous execution (like threads or processes).
    *   **Usage in this App:** The `ThreadPoolExecutor` is used within `TheDocProcessor.create_chunks` to parallelize the processing of chunks for large documents, potentially speeding up ingestion time.

*   **`re`:**
    *   **Purpose:** Python standard library for regular expression operations.
    *   **Usage in this App:** Used in `ContentSafetyChecker._pattern_check` for matching predefined harmful/sensitive patterns and in `TheDocProcessor` for basic text cleaning (e.g., normalizing whitespace) and attempting sentence boundary detection during chunking.

*   **Standard Libraries (`os`, `io`, `time`, `math`, `json`, `typing`, `numpy`, etc.):**
    *   **Purpose:** Provide fundamental functionalities.
    *   **Usage in this App:** Used for various tasks including: operating system interactions (`os.getenv`), handling file streams (`io.BytesIO`), measuring execution time (`time.time`), calculations like batch sizes (`math.ceil`), parsing LLM JSON responses (`json.loads`), defining function signatures and variable types (`typing`), and numerical operations (`numpy`, often implicitly via dependencies like scikit-learn/rank-bm25).


## 💾 Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher installed.
    *   Git (optional, for cloning).

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content (or generate it using `pip freeze > requirements.txt` if you installed manually):
    ```txt
    streamlit
    chromadb
    openai
    PyPDF2
    nltk
    rank_bm25
    python-dotenv
    numpy # Often a dependency of others, good to include
    scikit-learn # Dependency for rank-bm25 or potentially future use
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK Data:**
    The application attempts to download the necessary 'punkt' tokenizer data on first run if it's not found. You might need to run this manually once if issues occur:
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

6.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the project's root directory.
    *   Add your OpenAI API key to this file:
        ```
        OPENAI_API_KEY='your_openai_api_key_here'
        ```
    *   **Important:** Ensure this file is listed in your `.gitignore` if using version control to avoid committing your secret key.

## ▶️ How to Run

1.  **Activate your virtual environment** (if you created one):
    ```bash
    source venv/bin/activate # Or Windows equivalent
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run rag_query_system.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 🏗️ Code Structure Explained

The application is organized into several classes and a main Streamlit UI function:

1.  **`TheModelSelector`**:
    *   Purpose: Centralizes the selection of AI models (LLM and Embeddings). Currently hardcoded to use OpenAI models but designed for future extensibility.
    *   Methods: `get_models`, `get_embedding_info`.

2.  **`TheDocProcessor`**:
    *   Purpose: Handles all aspects of reading and chunking PDF documents.
    *   Key Methods:
        *   `read_pdf`: Reads text content from an uploaded PDF file, handling potential errors and showing progress for large files.
        *   `_process_pdf_chunk_internal`: Processes a single segment of text, applying overlap and attempting sentence boundary splitting.
        *   `create_chunks`: Orchestrates the chunking process, deciding between sequential and parallel execution based on document size. Returns a list of chunk dictionaries (id, text, metadata).

3.  **`ContentSafetyChecker`**:
    *   Purpose: Provides a layer of content moderation before adding documents.
    *   Key Methods:
        *   `_pattern_check`: Uses regular expressions (`re`) to find potentially problematic text patterns (e.g., harmful instructions).
        *   `_ai_moderation_check`: Uses the OpenAI Moderation API to analyze a text sample for various harmful content categories.
        *   `check_document`: Runs pattern checks and samples text for AI moderation, returning a list of detected issues (currently used to warn the user).

4.  **`QueryPlanner`**:
    *   Purpose: Analyzes user queries before retrieval.
    *   Key Methods:
        *   `determine_query_complexity`: Simple heuristic to guess if a query is complex based on length.
        *   `decompose_query`: If a query seems complex, uses an LLM (e.g., `gpt-4o-mini`) to try and break it down into simpler, actionable sub-queries.

5.  **`DocumentReranker`**:
    *   Purpose: Improves the relevance ordering of retrieved documents before they are sent to the LLM for answer generation.
    *   Key Methods:
        *   `rerank_with_bm25`: (Not used directly in the final stage but available) Reranks based on BM25 keyword scores.
        *   `rerank_with_cross_encoder`: Simulates a cross-encoder using an LLM. It prompts the LLM to score the relevance of each candidate document against the original query and returns the top-scoring documents.

6.  **`TheRAGSystem`**:
    *   Purpose: The core orchestrator of the RAG pipeline. Integrates all other components.
    *   Initialization (`__init__`): Sets up ChromaDB client, OpenAI clients (LLM, Embeddings), instantiates helper classes (`QueryPlanner`, `DocumentReranker`, `ContentSafetyChecker`), and prepares the ChromaDB collection. It also initializes an in-memory BM25 index.
    *   Key Methods:
        *   `_setup_collection`: Ensures the ChromaDB collection exists.
        *   `add_documents`: Takes chunks from `TheDocProcessor`, adds them to ChromaDB in batches, and triggers an update of the BM25 index.
        *   `_update_keyword_search_index_from_db`: Fetches all documents/metadata from ChromaDB and rebuilds the `rank-bm25` index in memory.
        *   `_keyword_search`: Performs a search using the in-memory BM25 index.
        *   `_hybrid_search`: Combines results from semantic search (ChromaDB `query`) and keyword search (`_keyword_search`), then uses `rerank_with_cross_encoder` for final ordering.
        *   `query_documents`: The main retrieval entry point. Handles optional query decomposition (`QueryPlanner`), calls the appropriate search method (hybrid or semantic), aggregates results from sub-queries (if any), and performs a final reranking based on the original query.
        *   `_check_hallucination`: Uses an LLM to compare the generated response against the source documents provided as context, identifying unsupported statements.
        *   `generate_response`: Takes the final set of context documents and the original query, formats a detailed prompt for the main LLM (instructing it to use only the context and follow a specific output structure), gets the response, optionally performs the hallucination check, and returns the final answer string (potentially with warnings/timing info).
        *   `get_collection_stats`: Queries ChromaDB for the number of chunks and unique source document names.
        *   `delete_document_by_source`: Finds all chunk IDs associated with a given source filename and deletes them from ChromaDB, then updates the BM25 index.

7.  **`main()` (Streamlit UI)**:
    *   Purpose: Defines and runs the Streamlit web interface.
    *   Structure: Uses `st.set_page_config`, `st.title`, session state (`st.session_state`) for persistence, sidebar for configuration/status, and tabs (`st.tabs`) for different sections (Chat, Upload, View Documents).
    *   Chat Tab: Manages chat history, displays user queries and formatted assistant responses (parsing the structured output, showing sources in an expander), handles user input, calls `TheRAGSystem.query_documents` and `TheRAGSystem.generate_response`, provides options for context filtering and temperature. Includes logic for deleting chat exchanges.
    *   Upload Tab: Provides a file uploader (`st.file_uploader`), options for chunking parameters, triggers the document processing pipeline (`TheDocProcessor`, `ContentSafetyChecker`, `TheRAGSystem.add_documents`) upon button click, shows progress and success/error messages.
    *   View Tab: Lists indexed documents retrieved via `TheRAGSystem.get_collection_stats`, provides a dropdown and button interface to trigger `TheRAGSystem.delete_document_by_source` with a confirmation step.

## 🔄 End-to-End Workflow

**1. Document Ingestion:**

1.  **Upload:** User selects a PDF file in the "Upload Documents" tab.
2.  **Processing Trigger:** User clicks "Process and Add Document".
3.  **Read:** `TheDocProcessor` reads the text content from the PDF.
4.  **Safety Check:** `ContentSafetyChecker` analyzes the text for potential issues (warnings shown if detected).
5.  **Chunking:** `TheDocProcessor` splits the text into overlapping chunks based on configured size/overlap, potentially using parallel processing.
6.  **Storage:** `TheRAGSystem.add_documents` adds the chunks (text, metadata, unique ID) to the ChromaDB collection. ChromaDB's `OpenAIEmbeddingFunction` automatically generates embeddings for each chunk text.
7.  **Keyword Index Update:** `TheRAGSystem._update_keyword_search_index_from_db` fetches the latest data from Chroma and rebuilds the in-memory BM25 index.
8.  **UI Update:** The UI refreshes (via `st.rerun`) to show the new document in the lists and updated stats.

**2. Querying:**

1.  **User Input:** User types a question into the chat input on the "Chat" tab and hits Enter/Submit. They can optionally select a specific document context and adjust the LLM temperature.
2.  **History Update:** The user's query is added to the `st.session_state.chat_history`. The UI reruns to display the query.
3.  **Processing Block:** The application detects a query needs processing.
4.  **Query Analysis (Optional):** `TheRAGSystem.query_documents` calls `QueryPlanner` to check complexity and potentially decompose the query into sub-queries.
5.  **Retrieval:** For each (sub-)query:
    *   `TheRAGSystem` calls `_hybrid_search` (default).
    *   `_hybrid_search` performs both a semantic query on ChromaDB and a keyword query using the BM25 index.
    *   Results are combined and initially ranked.
6.  **Consolidation & Reranking:**
    *   Unique document chunks retrieved across all sub-queries are collected.
    *   `DocumentReranker.rerank_with_cross_encoder` uses an LLM to re-score these candidate chunks against the *original* user query.
    *   The top N most relevant chunks (according to the reranker) are selected as the final context.
7.  **Response Generation:**
    *   `TheRAGSystem.generate_response` is called with the original query and the final context documents/metadata.
    *   A detailed prompt is constructed, instructing the main LLM (e.g., GPT-4o) to answer *only* based on the provided context and to use the specific structured format (DIRECT ANSWER, DETAILED EXPLANATION, KEY POINTS) with citations.
    *   The LLM generates the response.
8.  **Hallucination Check (Optional):** `_check_hallucination` uses another LLM call to verify if the generated response's claims are supported by the context documents.
9.  **Formatting & Storage:** Timing information is appended. The complete response string and the context used are stored in the `st.session_state.chat_history`.
10. **UI Display:** The UI reruns again. The chat history loop now finds the new assistant message. It parses the structured response (if possible), formats it using custom CSS, displays the source documents in an expander, and shows any hallucination warnings or timing info.

## ⚙️ Configuration

*   **OpenAI API Key:** The primary configuration required is your OpenAI API key. This *must* be placed in a `.env` file in the root directory of the project, formatted as:
    ```
    OPENAI_API_KEY='sk-...'
    ```
*   **Chunking Parameters:** Chunk size and overlap can be adjusted in the UI ("Upload Documents" tab -> Advanced Options) before processing a document.
*   **LLM Temperature:** Controls the randomness/creativity of the main LLM's response. Adjustable via a slider in the "Chat" tab.
*   **Collection Path:** The ChromaDB database is stored locally in the `./chroma_db` directory by default (defined in `TheRAGSystem`).

## 🚀 Potential Improvements & Future Work

*   **Support More File Types:** Extend `TheDocProcessor` to handle other formats like `.html`, `.md`, `.pptx`, etc (IF REQUIRED ONLY)
*   **Alternative Models:** Integrate support for other embedding or LLM providers (e.g., Hugging Face, Cohere, Anthropic) via `TheModelSelector`.
*   **More Sophisticated Safety:** Implement more robust content safety checks or allow configuration of safety levels (e.g., block vs. warn).
*   **UI Enhancements:** Improve chat history navigation, add search within history, visualize embedding space (e.g., with UMAP/t-SNE).
*   **Caching:** Implement caching for embeddings and potentially LLM responses to reduce costs and latency.
*   **Evaluation:** Add metrics and a framework for evaluating retrieval relevance and answer quality.
*   **Error Handling:** More granular error handling and user feedback throughout the pipeline.
*   **Asynchronous UI:** Make UI elements update more smoothly during long operations (though Streamlit's execution model makes this complex).
*   **Metadata Filtering:** Allow more complex filtering options during querying based on document metadata.