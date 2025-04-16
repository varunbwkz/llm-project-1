import streamlit as st
import logging
from utils import (
    get_openai_client,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_html,
    summarize_text_simple
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Session State ---
# Do this early, before using the keys
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
# ---> Add a key counter for the file uploader <---
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- Streamlit App UI ---

st.set_page_config(page_title="Simple Document Summarizer", layout="wide")
st.title("📄 Document Summarizer")
st.markdown("Upload a document (PDF, DOCX, TXT, HTML) to **automatically** generate a summary.")

# Initialize OpenAI client
@st.cache_resource
def initialize_openai():
    client = get_openai_client()
    if client is None:
        st.error("Failed to initialize OpenAI Client. Please check your API key in the .env file.")
    return client

client = initialize_openai()

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "txt", "html"],
    help="Supports PDF, Word (DOCX), Text (TXT), and HTML files.",
    # ---> Use the dynamic key <---
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# --- Processing Logic ---
if uploaded_file is not None:
    # Check if it's a new file being uploaded OR if it's the same filename but we haven't processed it yet this run
    # (This check handles the case after clearing where the filename might technically be the same briefly)
    if uploaded_file.name != st.session_state.get('current_file_name') or not st.session_state.processing_complete:
        st.info(f"Processing '{uploaded_file.name}'...")
        # Reset state for the new file (or first processing after clear)
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.extracted_text = None
        st.session_state.summary = None
        st.session_state.processing_complete = False # Explicitly mark as not complete yet

        file_content = uploaded_file.getvalue()
        file_type = uploaded_file.type
        extracted_text = None

        # 1. Extract Text
        with st.spinner("Extracting text from the document..."):
            try:
                if file_type == "application/pdf":
                    extracted_text = extract_text_from_pdf(file_content)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    extracted_text = extract_text_from_docx(file_content)
                elif file_type == "text/plain":
                    extracted_text = extract_text_from_txt(file_content)
                elif file_type == "text/html":
                    extracted_text = extract_text_from_html(file_content)
                else:
                    st.error(f"Unsupported file type: {file_type}")

                if extracted_text:
                    st.session_state.extracted_text = extracted_text
                    st.success("Text extracted successfully!")
                    logging.info(f"Extracted text length: {len(extracted_text)} characters.")
                elif extracted_text == "":
                     st.warning("Extracted text is empty. Cannot summarize.")
                     st.session_state.extracted_text = "" # Store empty string
                     st.session_state.processing_complete = True # Mark as done for this file
                else:
                    st.error("Failed to extract text. Cannot summarize.")
                    st.session_state.extracted_text = None # Indicate failure
                    st.session_state.processing_complete = True # Mark as done for this file

            except Exception as e:
                st.error(f"An error occurred during text extraction: {e}")
                logging.error(f"Extraction error for {uploaded_file.name}: {e}", exc_info=True)
                st.session_state.extracted_text = None
                st.session_state.processing_complete = True # Mark as done

        # 2. Generate Summary AUTOMATICALLY if text extraction was successful
        #    and processing for this file isn't already marked as complete
        if st.session_state.extracted_text and not st.session_state.processing_complete:
            if client:
                with st.spinner("Generating summary using OpenAI..."):
                    try:
                        summary = summarize_text_simple(st.session_state.extracted_text, client) # Use simple func
                        st.session_state.summary = summary
                        logging.info("Summary processing attempted.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during summarization trigger: {e}")
                        logging.error(f"Summarization trigger error: {e}", exc_info=True)
                        st.session_state.summary = None # Ensure summary is reset on error
            else:
                st.error("OpenAI client is not available. Cannot generate summary.")
            st.session_state.processing_complete = True # Mark processing as done for this file upload instance

# --- Display Results ---
# Show the summary only if it exists AND processing is marked complete for the current file
if st.session_state.processing_complete and st.session_state.summary:
    st.subheader("Generated Summary")
    if st.session_state.summary.startswith("Error:"):
         st.error(st.session_state.summary)
    else:
        st.markdown(st.session_state.summary)
# Handle cases where processing finished but there's no summary
elif st.session_state.processing_complete and not st.session_state.summary:
     if st.session_state.extracted_text == "":
         st.info("Document was empty or no text could be extracted.")
     else:
         # This case implies extraction worked, but summarization failed or returned nothing useful.
         # Error messages from summarize_text_simple are stored in st.session_state.summary
         # and handled by the block above. If summary is simply None/empty without Error:,
         # it means the API call might have failed silently or returned empty.
         st.warning("Summary could not be generated or was empty (check potential errors above or logs).")


# --- Clear Button ---
if st.button("Clear Upload and Summary"):
    # Clear the data state
    st.session_state.extracted_text = None
    st.session_state.summary = None
    st.session_state.current_file_name = None
    st.session_state.processing_complete = False
    # ---> Increment the key to force widget reset <---
    st.session_state.uploader_key += 1
    # Rerun the app
    st.rerun()