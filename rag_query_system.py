import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import uuid
import io
import time
import concurrent.futures
import math

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 4  # For parallel processing


class TheModelSelector:
    """I made this class to handle which AI models to use. Right now Im just using OpenAI
    because it works really well, but I might add more options later."""
    
    def __init__(self):
        # For now, I'm just using OpenAI for everything
        self.llm_model = "openai"
        self.embedding_model = "openai"
        
        # This is where I keep track of my embedding models and their settings
        self.embedding_models = {
            "openai": {
                "name": "OpenAI Embeddings",
                "dimensions": 1536,  # This is how detailed the AI's understanding of text is
                "model_name": "text-embedding-3-small",  # Using their newest model
            }
        }

    def get_models(self):
        """Just returns which models I'm using"""
        return self.llm_model, self.embedding_model


class ThePDFProcessor:
    """This is the PDF handler - it does all the work of reading PDFs and breaking them into
    smaller pieces that the AI can understand better"""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """This reads the PDF and shows a progress bar for big documents so users know it's working"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # I only show progress for big documents (more than 20 pages)
        total_pages = len(reader.pages)
        if total_pages > 20:
            progress_bar = st.progress(0)
            
        for i, page in enumerate(reader.pages):
            text += page.extract_text() + "\n"
            if total_pages > 20:
                progress_bar.progress((i + 1) / total_pages)
                
        if total_pages > 20:
            progress_bar.empty()
            
        return text

    def process_pdf_chunk(self, chunk_data):
        """This processes one piece of the PDF. I try to break at the end of sentences
        to keep things making sense"""
        start, end, text, pdf_name = chunk_data
        chunk = text[start:end]
        
        # Try to end chunks at sentence endings (periods)
        if end < len(text):
            last_period = chunk.rfind(".")
            if last_period != -1:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1
                
        return {
            "id": str(uuid.uuid4()),  # Give each chunk a unique ID
            "text": chunk,
            "metadata": {"source": pdf_name},  # Remember which PDF it came from
            "end_pos": end
        }

    def create_chunks(self, text, pdf_file):
        """This is where I split up the document into smaller pieces. For big documents,
        Im using parallel processing to make it faster"""
        
        # For smaller documents (less than 100KB), I keep it simple
        if len(text) < 100000:
            chunks = []
            start = 0

            while start < len(text):
                end = start + self.chunk_size
                if start > 0:
                    start = start - self.chunk_overlap

                chunk = text[start:end]

                # Try to break at sentence endings
                if end < len(text):
                    last_period = chunk.rfind(".")
                    if last_period != -1:
                        chunk = chunk[: last_period + 1]
                        end = start + last_period + 1

                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "metadata": {"source": pdf_file.name},
                    }
                )

                start = end
                
            return chunks
        else:
            # For big documents, I use parallel processing to speed things up
            chunks = []
            total_chunks = math.ceil(len(text) / self.chunk_size)
            chunk_data = []
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                if start > 0:
                    start = start - self.chunk_overlap
                    
                chunk_data.append((start, end, text, pdf_file.name))
                start = end
            
            # Show progress while processing
            progress_bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {executor.submit(self.process_pdf_chunk, cd): i for i, cd in enumerate(chunk_data)}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
                    try:
                        chunk_result = future.result()
                        chunks.append({
                            "id": chunk_result["id"],
                            "text": chunk_result["text"],
                            "metadata": chunk_result["metadata"]
                        })
                        progress_bar.progress((i + 1) / total_chunks)
                    except Exception as exc:
                        st.warning(f"Processing chunk failed: {exc}")
            
            progress_bar.empty()
            return chunks


class TheRAGSystem:
    """This is my main RAG system that ties everything together. It handles storing documents,
    searching through them, and generating answers to questions."""

    def __init__(self, embedding_model="openai", llm_model="openai"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Im using ChromaDB (the defacto standard from what I know...) to store and search through documents
        self.db = chromadb.PersistentClient(path="./chroma_db")

        # Setup the AI that understands text
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )

        # Setup the AI that answers questions
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Create or get (mainly GET) the document collection
        self.collection = self.setup_collection()

    def setup_collection(self):
        """This sets up where I store all the document pieces"""
        collection_name = "documents_openai"

        try:
            # Try to use existing collection first
            try:
                collection = self.db.get_collection(
                    name=collection_name, embedding_function=self.embedding_fn
                )
            except:
                # If it aint exist, then Im gonna make a new one... pray for me!!
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"model": "openai"},
                )
                st.success(
                    f"Created new collection for OpenAI embeddings"
                )

            return collection

        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e

    def add_documents(self, chunks):
        """This is where I add new document chunks to my database. I do it in batches
        for big documents to keep things running smoothly"""
        try:
            # Make sure we have a collection ready (fingers crossed)
            if not self.collection:
                self.collection = self.setup_collection()
                
            # For big sets of chunks, I process them in smaller batches
            batch_size = 100  # I found this size works well in general..
            
            if len(chunks) > batch_size:
                # Show progress as I add the batches
                total_batches = math.ceil(len(chunks) / batch_size)
                progress_bar = st.progress(0)
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    self.collection.add(
                        ids=[chunk["id"] for chunk in batch],
                        documents=[chunk["text"] for chunk in batch],
                        metadatas=[chunk["metadata"] for chunk in batch],
                    )
                    progress_bar.progress((i + batch_size) / len(chunks))
                
                progress_bar.empty()
            else:
                # For smaller sets, I just add them all at once
                self.collection.add(
                    ids=[chunk["id"] for chunk in chunks],
                    documents=[chunk["text"] for chunk in chunks],
                    metadatas=[chunk["metadata"] for chunk in chunks],
                )
                
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        """This is where I search through the documents to find relevant pieces
        that might answer the user's question"""
        try:
            if not self.collection:
                raise ValueError("No collection available")

            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def generate_response(self, query, context, temperature=0.7):
        """This is where the magic happens - I use the AI to create a helpful answer
        based on the relevant document pieces I found"""
        try:
            # First, I organize the context with source information
            context_with_sources = []
            
            for i, doc in enumerate(context):
                if isinstance(doc, dict) and "text" in doc and "metadata" in doc:
                    source = doc["metadata"].get("source", "Unknown")
                    context_with_sources.append(f"[Source: {source}] {doc['text']}")
                else:
                    context_with_sources.append(f"[Source {i+1}] {doc}")
                    
            formatted_context = "\n\n".join(context_with_sources)
            
            # I made this prompt really clear so the AI knows exactly what I want
            prompt = f"""
            You are a knowledgeable assistant tasked with providing answers STRICTLY based on the provided document excerpts. 
            
            IMPORTANT INSTRUCTIONS:
            - ONLY use information found in the provided document excerpts to answer
            - If the document excerpts don't contain information needed to answer the question, respond with "I don't have enough information to answer this question based on the provided documents."
            - Do NOT use your general knowledge to fill gaps in the documents
            - Be extremely careful not to hallucinate information
            - If the documents are completely unrelated to the query, state this clearly
            - NEVER make up information that's not in the documents
            
            Question: {query}

            Below are the only document excerpts you can use to answer:
            {formatted_context}

            If you can answer the question, provide a detailed response that:
            1. Directly answers the question using only information from the documents
            2. Synthesizes information from multiple sources when relevant
            3. Highlights any contradictions between sources if they exist
            4. Uses quoted snippets from the sources when particularly relevant

            Format your response as follows:

            DIRECT ANSWER:
            [Concise answer addressing the question directly OR state that you don't have enough information]

            DETAILED EXPLANATION:
            [Detailed explanation with supporting evidence from the sources OR explanation of why the documents don't contain relevant information]

            KEY POINTS:
            - [Point 1 with source reference]
            - [Point 2 with source reference]
            - [Point 3 with source reference]

            Answer:
            """

            # Get the AI's response
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that ONLY provides answers based on the given context. You must refuse to answer questions if the information is not in the provided documents."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature
            )

            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self):
        """This just tells me what embedding model I'm using and its settings"""
        model_selector = TheModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }
        
    def get_collection_stats(self):
        """This gives me stats about my document collection - how many chunks I have
        and which documents they came from"""
        try:
            if not self.collection:
                return {"count": 0, "sources": []}
                
            # Count how many chunks I have
            all_items = self.collection.get()
            count = len(all_items["ids"]) if "ids" in all_items else 0
            
            # Get a list of unique document sources
            sources = set()
            if "metadatas" in all_items and all_items["metadatas"]:
                for metadata in all_items["metadatas"]:
                    if metadata and "source" in metadata:
                        sources.add(metadata["source"])
                        
            return {
                "count": count,
                "sources": sorted(list(sources))
            }
        except Exception as e:
            st.error(f"Error getting collection stats: {str(e)}")
            return {"count": 0, "sources": []}
            
    def delete_document_by_source(self, source_name):
        """This helps me remove a specific document and all its chunks from my database"""
        try:
            if not self.collection:
                return False
                
            # First, get everything in my collection
            all_items = self.collection.get()
            
            if not all_items or "ids" not in all_items or not all_items["ids"]:
                return False
                
            # Find all the chunks that came from this document
            ids_to_delete = []
            
            for i, metadata in enumerate(all_items["metadatas"]):
                if metadata and "source" in metadata and metadata["source"] == source_name:
                    ids_to_delete.append(all_items["ids"][i])
                    
            # If I found chunks to delete, remove them
            if ids_to_delete:
                # Let the user know what I'm doing
                st.info(f"Deleting {len(ids_to_delete)} chunks from source: {source_name}")
                
                # Delete the chunks
                self.collection.delete(ids=ids_to_delete)
                
                # Double check that they're really gone
                verification = self.collection.get()
                remaining_ids = []
                for i, metadata in enumerate(verification["metadatas"]):
                    if metadata and "source" in metadata and metadata["source"] == source_name:
                        remaining_ids.append(verification["ids"][i])
                
                if remaining_ids:
                    st.warning(f"Warning: {len(remaining_ids)} chunks from {source_name} still remain in the database.")
                    return False
                else:
                    return True
                
            return False
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False
            
    def reset_collection(self):
        """This is my emergency reset button - it deletes everything from the database"""
        try:
            if not self.collection:
                return False
                
            # Get everything in the collection
            all_items = self.collection.get()
            
            if all_items and "ids" in all_items and all_items["ids"]:
                # Delete it all
                self.collection.delete(ids=all_items["ids"])
                st.success(f"Successfully reset collection. Removed {len(all_items['ids'])} chunks.")
                return True
            return False
        except Exception as e:
            st.error(f"Error resetting collection: {str(e)}")
            return False


def main():
    st.title("🤖 Enhanced RAG System")

    # I keep track of important stuff between page refreshes using session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()  # Remember which files I've processed
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = "openai"  # Using OpenAI by default
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0  # This helps me refresh the upload widget
    if "query_history" not in st.session_state:
        st.session_state.query_history = []  # Keep track of previous questions and answers
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "current_response" not in st.session_state:
        st.session_state.current_response = None

    # Set up my AI models
    model_selector = TheModelSelector()
    llm_model, embedding_model = model_selector.get_models()

    # Initialize my RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = TheRAGSystem(embedding_model, llm_model)

        # Show some stats in the sidebar
        stats = st.session_state.rag_system.get_collection_stats()
        if stats["count"] > 0:
            st.sidebar.success(f"📊 Collection Stats: {stats['count']} chunks from {len(stats['sources'])} documents")
            
        # Show which AI models I'm using
        embedding_info = model_selector.embedding_models["openai"]
        st.sidebar.info(
            f"📚 Using OpenAI Models:\n"
            f"- LLM: GPT-4\n"
            f"- Embeddings: {embedding_info['name']}\n"
            f"- Dimensions: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # Create my two main tabs - one for uploading, one for asking questions
    tab1, tab2 = st.tabs(["📄 Document Upload", "🔍 Query Documents"])
    
    with tab1:
        st.subheader("Upload Documents")
        
        # Show the document management section if there are documents
        stats = st.session_state.rag_system.get_collection_stats()
        source_list = stats.get("sources", [])
        if source_list:
            st.success(f"You have {len(source_list)} documents uploaded with {stats['count']} total chunks.")
            
            # Add a section for removing documents
            st.subheader("🗑️ Remove Documents")
            
            # Make it look nice with columns
            col1, col2 = st.columns([4, 1])
            
            with col1:
                doc_to_delete = st.selectbox(
                    "Select document to remove:", 
                    options=source_list,
                    key="doc_delete_select"
                )
            
            with col2:
                # Add some spacing to align the button
                st.markdown("<div style='padding-top: 25px;'></div>", unsafe_allow_html=True)
                if st.button("Remove", type="secondary", use_container_width=True):
                    if doc_to_delete:
                        with st.spinner(f"Removing {doc_to_delete} from database..."):
                            if st.session_state.rag_system.delete_document_by_source(doc_to_delete):
                                if doc_to_delete in st.session_state.processed_files:
                                    st.session_state.processed_files.remove(doc_to_delete)
                                st.success(f"Successfully removed {doc_to_delete} from the database!")
                                
                                st.session_state.upload_key += 1
                                st.rerun()
                            else:
                                st.error(f"Failed to remove {doc_to_delete}. Document not found in database.")
            
            st.markdown("---")
        
        st.write("Upload one or more PDF documents to process and add to the knowledge base. Once done, you can query for any information from the uploaded documents in the 'Query Documents' tab.")
        
        # My file upload widget
        pdf_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True, key=f"uploader_{st.session_state.upload_key}")

        # Advanced settings that users can adjust
        with st.expander("Advanced Options"):
            chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=CHUNK_SIZE, step=100, 
                                help="Think of this like breaking a big book into smaller sections. A larger chunk size means bigger sections (more context but might be less precise), while smaller chunks mean shorter sections (more precise but might miss context). Example: with 1000, a page might be split into 3-4 parts.")
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=CHUNK_OVERLAP, step=50,
                                    help="This is like having each section share a few sentences with the next section, so we don't lose the connection between ideas. Example: if overlap is 200, each chunk will share about 200 characters with the next chunk to maintain context.")
        
        # Show the list of uploaded documents
        if source_list:
            st.subheader("📚 List of Uploaded Documents")
            for i, source in enumerate(source_list, 1):
                st.markdown(f"**{i}. {source}**")

        # Process any new uploaded files
        if pdf_files:
            processor = ThePDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            for pdf_file in pdf_files:
                if pdf_file.name not in st.session_state.processed_files:
                    with st.spinner(f"Processing {pdf_file.name}..."):
                        try:
                            # Check the file size
                            pdf_file.seek(0, os.SEEK_END)
                            file_size = pdf_file.tell()
                            pdf_file.seek(0)
                            
                            size_mb = file_size / (1024 * 1024)
                            st.info(f"Processing {pdf_file.name} ({size_mb:.2f} MB)")
                            
                            start_time = time.time()
                            
                            # Process the PDF
                            text = processor.read_pdf(pdf_file)
                            chunks = processor.create_chunks(text, pdf_file)
                            
                            # Add the chunks to my database
                            if st.session_state.rag_system.add_documents(chunks):
                                st.session_state.processed_files.add(pdf_file.name)
                                
                                end_time = time.time()
                                st.success(f"✅ Successfully processed {pdf_file.name} ({len(chunks)} chunks in {end_time - start_time:.2f} seconds)")
                        except Exception as e:
                            st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
            
            # Refresh the page to show the new documents
            st.session_state.upload_key += 1
            st.rerun()
    
    with tab2:
        # This is my question-answering interface
        if stats["count"] > 0:
            # Make it look nice
            st.markdown(
                """
                <style>
                div.stButton > button {
                    font-weight: 500;
                }
                div.stButton > button:focus {
                    box-shadow: none;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Create a nice header with a clear button
            header_col1, header_col2 = st.columns([4, 1])
            
            with header_col1:
                st.markdown("<h3 style='margin-bottom:0.8rem;'>Query Your Documents</h3>", unsafe_allow_html=True)
                
            with header_col2:
                st.markdown("<div style='padding-top: 8px;'></div>", unsafe_allow_html=True)
                if st.button("🧹 Clear", key="clear_btn", type="secondary", use_container_width=True):
                    st.session_state.current_query = ""
                    st.session_state.current_response = None
                    st.session_state.query_history = []
                    st.rerun()
            
            st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)
            
            # Create my main query interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query = st.text_area("Enter your question:", 
                                    value=st.session_state.current_query,
                                    height=100, 
                                    placeholder="Ask a question about the uploaded documents...")
                if query != st.session_state.current_query:
                    st.session_state.current_query = query
            
            with col2:
                st.write("Options:")
                n_results = st.slider("Passages to retrieve:", min_value=1, max_value=10, value=3, 
                                    help="Number of relevant passages to retrieve from the documents.")
                
                temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                       help="Think of temperature like a creativity dial. At 0.0 (cold), the AI gives consistent, focused answers - great for factual queries. At 1.0 (hot), it gets more creative and varied - better for brainstorming. Example: For 'What is quantum computing?', 0.2 gives technical definitions, while 0.8 might include analogies and examples.")

            # Process the question when the user clicks submit
            if st.button("🔍 Submit Query", type="primary"):
                if query:
                    with st.spinner("Generating response..."):
                        # Find relevant document pieces
                        results = st.session_state.rag_system.query_documents(query, n_results=n_results)
                        
                        if results and results["documents"] and results["documents"][0]:
                            # Generate the answer
                            response = st.session_state.rag_system.generate_response(
                                query, results["documents"][0], temperature=temperature
                            )

                            if response:
                                # Save everything for history
                                st.session_state.current_response = {
                                    "query": query,
                                    "response": response,
                                    "sources": results["documents"][0],
                                    "metadatas": results["metadatas"][0]
                                }
                                
                                st.session_state.query_history.append(st.session_state.current_response)
                                
                                st.rerun()
                        else:
                            st.error("No relevant information found in the documents. Try rephrasing your question.")
                else:
                    st.warning("Please enter a query first.")
            
            # Show the answer if we have one
            if st.session_state.current_response:
                # Make it look pretty with custom CSS
                st.markdown(
                    """
                    <style>
                    .answer-container {
                        background-color: #1E1E1E;
                        border: 1px solid #333;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 10px 0;
                    }
                    .answer-section {
                        margin-bottom: 20px;
                    }
                    .answer-section:last-child {
                        margin-bottom: 0;
                    }
                    .section-title {
                        color: #FF6B6B;
                        font-size: 1.1em;
                        font-weight: 600;
                        margin-bottom: 10px;
                    }
                    .key-point {
                        background-color: #2A2A2A;
                        border-left: 3px solid #FF6B6B;
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 0 5px 5px 0;
                    }
                    .source-tag {
                        background-color: #2E4057;
                        color: #B8C9E8;
                        padding: 2px 6px;
                        border-radius: 4px;
                        font-size: 0.9em;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Show the answer in a nice container
                st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                
                # Direct Answer Section
                st.markdown('<div class="answer-section">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📌 Direct Answer</div>', unsafe_allow_html=True)
                try:
                    if "DIRECT ANSWER:" in st.session_state.current_response["response"]:
                        direct_answer = st.session_state.current_response["response"].split("DIRECT ANSWER:")[1].split("DETAILED EXPLANATION:")[0].strip()
                    else:
                        # Fallback if format is different
                        direct_answer = st.session_state.current_response["response"].split("\n\n")[0].strip()
                except IndexError:
                    direct_answer = st.session_state.current_response["response"].strip()
                st.markdown(f"{direct_answer}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed Explanation Section
                st.markdown('<div class="answer-section">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📝 Detailed Explanation</div>', unsafe_allow_html=True)
                try:
                    if "DETAILED EXPLANATION:" in st.session_state.current_response["response"] and "KEY POINTS:" in st.session_state.current_response["response"]:
                        detailed_explanation = st.session_state.current_response["response"].split("DETAILED EXPLANATION:")[1].split("KEY POINTS:")[0].strip()
                    elif "DETAILED EXPLANATION:" in st.session_state.current_response["response"]:
                        # If KEY POINTS section is missing
                        detailed_explanation = st.session_state.current_response["response"].split("DETAILED EXPLANATION:")[1].strip()
                    else:
                        # Fallback if format is different
                        parts = st.session_state.current_response["response"].split("\n\n")
                        detailed_explanation = "\n\n".join(parts[1:-1]) if len(parts) > 2 else st.session_state.current_response["response"]
                except IndexError:
                    detailed_explanation = "Error parsing the detailed explanation section."
                st.markdown(f"{detailed_explanation}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Key Points Section
                st.markdown('<div class="answer-section">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🔑 Key Points</div>', unsafe_allow_html=True)
                try:
                    if "KEY POINTS:" in st.session_state.current_response["response"]:
                        key_points = st.session_state.current_response["response"].split("KEY POINTS:")[1].strip()
                        for point in key_points.split("\n"):
                            if point.strip() and point.strip() != "-":
                                st.markdown(f'<div class="key-point">{point.strip().replace("- ", "")}</div>', unsafe_allow_html=True)
                    else:
                        # Fallback if KEY POINTS section is missing
                        st.markdown('<div class="key-point">Key points not provided in this response format.</div>', unsafe_allow_html=True)
                except IndexError:
                    st.markdown('<div class="key-point">Error parsing the key points section.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Show where the information came from
                with st.expander("🔍 View Source Passages", expanded=False):
                    for idx, (doc, metadata) in enumerate(zip(
                            st.session_state.current_response["sources"], 
                            st.session_state.current_response["metadatas"]), 1):
                        source = metadata.get("source", "Unknown")
                        st.markdown(f'<div class="source-tag">Source {idx}: {source}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="margin: 10px 0 20px 0; padding: 15px; background-color: #1E1E1E; border-radius: 5px;">{doc}</div>', unsafe_allow_html=True)
            
            # Show previous questions and answers
            if len(st.session_state.query_history) > 1:
                with st.expander("Previous Queries", expanded=False):
                    for i, item in enumerate(st.session_state.query_history[:-1]):
                        st.markdown(f"#### Query {i+1}: {item['query']}")
                        st.markdown(item['response'])
                        st.markdown("---")
        else:
            st.info("👆 Please upload documents in the 'Document Upload' tab first!")


if __name__ == "__main__":
    main()
