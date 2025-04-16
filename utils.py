import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError # Import specific error
import PyPDF2
import docx
from bs4 import BeautifulSoup
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment and API Setup ---
def load_api_key():
    """Loads OpenAI API key from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
    return api_key

def get_openai_client():
    """Initializes and returns the OpenAI client."""
    try:
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)
        return client
    except ValueError as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during OpenAI client initialization: {e}")
        return None

# --- Text Extraction Functions (Keep these as they are essential) ---
def extract_text_from_pdf(file_content):
    """Extracts text from PDF file content (bytes)."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logging.info(f"Extracted {len(text)} characters from PDF.")
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(file_content):
    """Extracts text from DOCX file content (bytes)."""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        logging.info(f"Extracted {len(text)} characters from DOCX.")
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        return None

def extract_text_from_txt(file_content):
    """Extracts text from TXT file content (bytes)."""
    try:
        text = file_content.decode('utf-8', errors='ignore')
        logging.info(f"Extracted {len(text)} characters from TXT.")
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from TXT: {e}")
        return None

def extract_text_from_html(file_content):
    """Extracts text from HTML file content (bytes)."""
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        logging.info(f"Extracted {len(text)} characters from HTML.")
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from HTML: {e}")
        return None

# --- Simplified OpenAI Summarization Logic ---

SUMMARY_MODEL = "gpt-4o-mini" # Or "gpt-4-turbo-preview", "gpt-4o-mini" etc.
# Note: Check model context limits. gpt-3.5-turbo usually 4k or 16k tokens.
# gpt-4-turbo-preview has 128k tokens.

SUMMARY_PROMPT_TEMPLATE = """
Please provide a detailed and precise summary of the following document. Capture the main points, key arguments, and conclusions. Keep the summary concise but informative.

Document Text:
"{document_text}"

Detailed and Precise Summary:
"""

def summarize_text_simple(text, client):
    """
    Summarizes the text using a SINGLE OpenAI API call.
    WARNING: This will fail if the text exceeds the model's context limit.

    Args:
        text (str): The text to summarize.
        client (OpenAI): The initialized OpenAI client.

    Returns:
        str: The generated summary, or an error message.
    """
    if not client:
        return "Error: OpenAI client not initialized."
    if not text:
        return "Error: No text provided to summarize."

    logging.info(f"Attempting direct summarization of {len(text)} characters...")

    try:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(document_text=text)

        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert assistant skilled in summarizing documents accurately and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000 # Adjust max tokens for summary length as needed
        )
        summary = response.choices[0].message.content.strip()
        logging.info("Summary generated successfully (simple method).")
        return summary

    # Catch the specific error related to context length
    except OpenAIError as e:
         if hasattr(e, 'code') and e.code == 'context_length_exceeded':
             error_msg = (f"Error: The document is too long for the '{SUMMARY_MODEL}' model's context window. "
                         "Please use a shorter document or implement document chunking for longer texts.")
             logging.error(error_msg)
             return error_msg
         else:
             # Handle other potential API errors
             error_msg = f"An OpenAI API error occurred: {e}"
             logging.error(error_msg, exc_info=True)
             return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during summarization: {e}"
        logging.error(error_msg, exc_info=True)
        return f"An unexpected error occurred: {e}"