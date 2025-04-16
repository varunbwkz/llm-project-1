import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
# import PyPDF2 # No longer needed
# import docx # No longer needed
import uuid
import io
import time
import concurrent.futures
import math
import json
import re
from typing import Union, Optional, Dict, List, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import traceback # For detailed error logging
import requests # Added for URL fetching
import trafilatura # Added for robust text extraction

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 4
URL_CONFIG_FILE = "knowledge_urls.txt"
REQUEST_TIMEOUT = 20
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- Model Selection (Unchanged) ---
class TheModelSelector:
    def __init__(self):
        self.llm_model = "openai"
        self.embedding_model = "openai"
        self.embedding_models = {"openai": {"name": "OpenAI Embeddings (text-embedding-3-small)", "dimensions": 1536, "model_name": "text-embedding-3-small"}}
        self.llm_models = {"openai": {"qa_model": "gpt-4o-mini", "hallucination_check_model": "gpt-4o-mini"}}
    def get_models(self) -> Tuple[str, str]: return self.llm_model, self.embedding_model
    def get_embedding_info(self, model_key: str = "openai") -> Optional[Dict[str, Any]]: return self.embedding_models.get(model_key)
    def get_llm_info(self, model_key: str = "openai") -> Optional[Dict[str, Any]]: return self.llm_models.get(model_key)

# --- URL Content Fetching and Processing ---
class TheURLProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
             print(f"Warning: Chunk overlap ({self.chunk_overlap}) >= chunk size ({self.chunk_size}). Setting overlap to {self.chunk_size // 5}.")
             self.chunk_overlap = self.chunk_size // 5

    def fetch_and_extract_text(self, url: str) -> Optional[str]:
        print(f"Fetching content from: {url}")
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            html_content = response.text
            extracted_text = trafilatura.extract(html_content, include_comments=False, include_tables=True, favor_recall=True)
            if not extracted_text:
                print(f"Warning: Trafilatura couldn't extract main content from {url}. Trying basic extraction.")
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                extracted_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
            if not extracted_text: print(f"Warning: No significant text content found at {url}"); return None
            cleaned_text = re.sub(r'\s+', ' ', extracted_text).strip()
            print(f"  - Extracted ~{len(cleaned_text)} characters.")
            return cleaned_text
        except requests.exceptions.RequestException as e: print(f"Error fetching URL {url}: {e}"); st.warning(f"Could not fetch content from URL: {url}. Error: {e}", icon="🌐"); return None
        except Exception as e: print(f"Error processing URL {url}: {e}"); st.warning(f"Could not process content from URL: {url}. Error: {e}", icon="⚙️"); traceback.print_exc(); return None

    # <<< CHANGE >>> Accept display_name
    def _process_doc_chunk_internal(self, chunk_data: Tuple[int, int, str, str, str]) -> Dict[str, Any]:
        start, end, text, source_url, display_name = chunk_data # <<< ADDED display_name
        effective_start = max(0, start - self.chunk_overlap) if start > 0 else 0
        chunk_text = text[effective_start:end]
        final_end = end
        if end < len(text):
            search_start = max(0, len(chunk_text) - self.chunk_overlap - 50)
            possible_ends = [m.start() + 1 for m in re.finditer(r'[.!?]\s', chunk_text[search_start:])]
            if possible_ends:
                ideal_end_in_chunk = end - effective_start
                best_end_in_chunk = -1; min_diff = float('inf')
                for p_end_relative in possible_ends:
                    p_end_absolute_in_chunk = search_start + p_end_relative
                    if abs(p_end_absolute_in_chunk - ideal_end_in_chunk) < self.chunk_size * 0.4:
                         diff = abs(p_end_absolute_in_chunk - ideal_end_in_chunk)
                         if diff < min_diff: min_diff = diff; best_end_in_chunk = p_end_absolute_in_chunk
                if best_end_in_chunk != -1: final_end = effective_start + best_end_in_chunk; chunk_text = text[effective_start:final_end]
        else: final_end = len(text); chunk_text = text[effective_start:final_end]
        chunk_text = chunk_text.strip()
        return {
            "id": str(uuid.uuid4()),
            "text": chunk_text,
            # <<< CHANGE >>> Store both URL (source) and Display Name
            "metadata": {"source": source_url, "display_name": display_name, "start": effective_start, "end": final_end},
            "end_pos": final_end
        }

    # <<< CHANGE >>> Accept display_name
    def create_chunks(self, text: str, source_url: str, display_name: str) -> List[Dict[str, Any]]:
        if not text: return []
        chunks = []; doc_len = len(text)
        use_parallel = doc_len > 150000 and MAX_WORKERS > 1

        if not use_parallel:
            print(f"Using sequential chunking for '{display_name}' ({source_url})...")
            current_pos = 0
            while current_pos < doc_len:
                end = min(current_pos + self.chunk_size, doc_len)
                chunk_data = (current_pos, end, text, source_url, display_name) # <<< ADDED display_name
                processed_chunk = self._process_doc_chunk_internal(chunk_data)
                if processed_chunk["text"]: chunks.append({"id": processed_chunk["id"], "text": processed_chunk["text"], "metadata": processed_chunk["metadata"]})
                next_start = processed_chunk["end_pos"]
                if next_start <= current_pos: print(f"Warning: Chunking stalled at position {current_pos} for {display_name}. Advancing."); next_start = current_pos + 1
                current_pos = next_start
        else:
            print(f"Using parallel chunking for '{display_name}' ({source_url}) (Length: {doc_len} chars)...")
            tasks = []
            current_pos = 0
            while current_pos < doc_len:
                 end = min(current_pos + self.chunk_size, doc_len)
                 tasks.append((current_pos, end, text, source_url, display_name)) # <<< ADDED display_name
                 current_pos = end
            results = []
            print(f"Chunking '{display_name}'... ({len(tasks)} tasks)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {executor.submit(self._process_doc_chunk_internal, task): task for task in tasks}
                count_completed = 0
                for future in concurrent.futures.as_completed(future_to_task):
                    task_data = future_to_task[future]
                    try:
                        result = future.result(); results.append(result); count_completed += 1
                        if count_completed % 10 == 0: print(f"  Chunking progress for {display_name}: {count_completed}/{len(tasks)} tasks complete.")
                    except Exception as exc: print(f"Chunk generation failed for {display_name} part starting near {task_data[0]}: {exc}")
            results.sort(key=lambda r: r['metadata']['start'])
            chunks = [{"id": r["id"], "text": r["text"], "metadata": r["metadata"]} for r in results if r["text"]]

        print(f"Generated {len(chunks)} chunks for '{display_name}'.")
        return chunks

# --- RAG System (Adapted) ---
class TheRAGSystem:
    def __init__(self, embedding_model_provider="openai", llm_provider="openai"):
        self.embedding_model_provider = embedding_model_provider
        self.llm_provider = llm_provider
        self.db_path = "./chroma_db_urls"
        self.collection_name = f"url_docs_{self.embedding_model_provider}"
        self.model_selector = TheModelSelector()
        self.llm_info = self.model_selector.get_llm_info(self.llm_provider)
        self.emb_info = self.model_selector.get_embedding_info(self.embedding_model_provider)
        if not self.llm_info or not self.emb_info: st.error("Failed to load model configurations."); print("ERROR: Failed to load model configurations."); st.stop()
        try: self.db = chromadb.PersistentClient(path=self.db_path)
        except Exception as e: st.error(f"Fatal Error: Could not initialize ChromaDB client at '{self.db_path}': {e}"); print(f"FATAL: Could not initialize ChromaDB client at '{self.db_path}': {e}"); st.stop()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key: st.error("Fatal Error: OPENAI_API_KEY environment variable not found."); print("FATAL: OPENAI_API_KEY environment variable not found."); st.stop()
        try:
            if self.embedding_model_provider == "openai": self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=self.openai_api_key, model_name=self.emb_info['model_name'])
            else: raise NotImplementedError(f"Embedding provider '{self.embedding_model_provider}' not supported yet.")
        except Exception as e: st.error(f"Fatal Error: Could not initialize embedding function: {e}"); print(f"FATAL: Could not initialize embedding function: {e}"); st.stop()
        try:
            if self.llm_provider == "openai": self.llm = OpenAI(api_key=self.openai_api_key)
            else: raise NotImplementedError(f"LLM provider '{self.llm_provider}' not supported yet.")
        except Exception as e: st.error(f"Fatal Error: Could not initialize OpenAI LLM client: {e}"); print(f"FATAL: Could not initialize OpenAI LLM client: {e}"); self.llm = None; st.stop()
        self.collection = self._setup_collection()
        self.corpus: List[str] = []; self.doc_ids: List[str] = []; self.doc_metadatas: List[Dict] = []; self.bm25: Optional[BM25Okapi] = None
        # <<< CHANGE >>> Add mapping for UI lookups
        self.display_name_to_url: Dict[str, str] = {}
        self.url_to_display_name: Dict[str, str] = {}


    def _setup_collection(self) -> Optional[chromadb.Collection]:
        try:
            collection = self.db.get_or_create_collection(name=self.collection_name, embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}) # type: ignore [arg-type]
            print(f"Successfully accessed or created collection: '{self.collection_name}'")
            return collection
        except Exception as e: st.error(f"Fatal Error: Could not get or create ChromaDB collection '{self.collection_name}': {e}"); print(f"Collection setup error for '{self.collection_name}':"); traceback.print_exc(); st.stop(); return None

    def _update_keyword_search_index_from_db(self):
        print("Attempting to update keyword (BM25) index...")
        if not self.collection: print("Warning: BM25 update skipped, collection not available."); return
        try:
            all_data = self.collection.get(include=["metadatas", "documents"])
            if not all_data or not all_data.get("ids"):
                print("Keyword index reset: No documents found in the collection."); self.corpus = []; self.doc_ids = []; self.doc_metadatas = []; self.bm25 = None; return
            self.doc_ids = all_data["ids"]
            self.corpus = all_data["documents"] or []
            self.doc_metadatas = all_data["metadatas"] or []
            if not self.corpus: print("Warning: No document text found for BM25 index, clearing index."); self.bm25 = None; return
            print(f"Building BM25 index for {len(self.corpus)} document chunks...")
            start_time = time.time()
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            end_time = time.time(); print(f"BM25 index built successfully in {end_time - start_time:.2f} seconds.")

            # <<< CHANGE >>> Populate mappings after index is built/updated
            self._update_url_mappings()

        except Exception as e: st.error(f"Error updating keyword search index: {e}"); print(f"Keyword (BM25) index update failed:"); traceback.print_exc(); self.bm25 = None

    # <<< CHANGE >>> New method to update the display name <-> URL mappings
    def _update_url_mappings(self):
        """Updates the internal dictionaries mapping display names to URLs and vice versa."""
        self.display_name_to_url = {}
        self.url_to_display_name = {}
        if not self.doc_metadatas:
             print("No metadata found to build URL mappings.")
             return

        unique_sources = set()
        for meta in self.doc_metadatas:
            if meta and "source" in meta and "display_name" in meta:
                 url = meta["source"]
                 name = meta["display_name"]
                 # Only add if we haven't seen this specific URL before to get unique pairs
                 if url not in unique_sources:
                     if name in self.display_name_to_url and self.display_name_to_url[name] != url:
                          print(f"Warning: Duplicate display name '{name}' detected for different URLs ('{url}' and '{self.display_name_to_url[name]}'). Filtering might be ambiguous.")
                          st.warning(f"Duplicate source name '{name}' detected. Filtering by this name might yield results from multiple URLs.", icon="⚠️")
                     self.display_name_to_url[name] = url
                     self.url_to_display_name[url] = name
                     unique_sources.add(url)

        print(f"Updated URL mappings: {len(self.display_name_to_url)} unique sources found.")


    def _keyword_search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> Dict[str, List]:
        if not self.bm25 or not self.corpus or not self.doc_ids: return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        try:
            tokens = query.lower().split()
            if not tokens: return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            scores = self.bm25.get_scores(tokens)
            results_with_scores = []
            for i, score in enumerate(scores):
                if i >= len(self.doc_metadatas) or i >= len(self.doc_ids) or i >= len(self.corpus): print(f"Warning: Index mismatch during keyword search at index {i}."); continue
                metadata, doc_id, document_text = self.doc_metadatas[i], self.doc_ids[i], self.corpus[i]
                if where_filter:
                    match = True
                    for key, value in where_filter.items():
                        # <<< CHANGE >>> Check metadata field directly
                        if str(metadata.get(key)) != str(value): match = False; break
                    if not match: continue
                if score > 0: results_with_scores.append((i, doc_id, score, metadata, document_text))
            results_with_scores.sort(key=lambda x: x[2], reverse=True)
            top_results = results_with_scores[:n_results]
            final_ids = [r[1] for r in top_results]; final_docs = [r[4] for r in top_results]; final_metas = [r[3] for r in top_results]; final_dists = []
            if top_results:
                scores_only = [r[2] for r in top_results]; max_score, min_score = (max(scores_only), min(scores_only)) if scores_only else (1.0, 0.0)
                score_range = max_score - min_score if max_score > min_score else 1.0
                final_dists = [max(0.0, 1.0 - ((r[2] - min_score) / score_range)) if score_range > 0 else 0.5 for r in top_results]
            return {"ids": [final_ids], "documents": [final_docs], "metadatas": [final_metas], "distances": [final_dists]}
        except Exception as e: st.warning(f"Keyword search failed: {e}"); print(f"Keyword search error:"); traceback.print_exc(); return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _hybrid_search(self, query: str, n_results: int = 5, where_filter: Optional[Dict] = None) -> Dict[str, List]:
        print(f"Performing Hybrid Search (RRF) for query: '{query[:50]}...' Filter: {where_filter}")
        if not self.collection: st.error("Hybrid search failed: Collection is not available."); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
        try:
            semantic_n = max(n_results * 2, 10)
            semantic_results = self.collection.query(query_texts=[query], n_results=semantic_n, where=where_filter, include=["metadatas", "documents", "distances"])
            keyword_n = max(n_results * 2, 10)
            keyword_results = self._keyword_search(query, n_results=keyword_n, where_filter=where_filter)
            k = 60; fused_scores: Dict[str, float] = {}; doc_details: Dict[str, Dict[str, Any]] = {}
            if semantic_results and semantic_results.get("ids") and semantic_results["ids"][0]:
                sem_ids, sem_docs, sem_metas = semantic_results["ids"][0], semantic_results["documents"][0], semantic_results["metadatas"][0]
                for rank, doc_id in enumerate(sem_ids):
                    if doc_id: score = 1 / (k + rank + 1); fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score
                    if doc_id not in doc_details: doc_text = sem_docs[rank] if rank < len(sem_docs) else "N/A"; doc_meta = sem_metas[rank] if rank < len(sem_metas) else {}; doc_details[doc_id] = {"doc": doc_text, "meta": doc_meta}
            if keyword_results and keyword_results.get("ids") and keyword_results["ids"][0]:
                key_ids, key_docs, key_metas = keyword_results["ids"][0], keyword_results["documents"][0], keyword_results["metadatas"][0]
                for rank, doc_id in enumerate(key_ids):
                     if doc_id: score = 1 / (k + rank + 1); fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score
                     if doc_id not in doc_details: doc_text = key_docs[rank] if rank < len(key_docs) else "N/A"; doc_meta = key_metas[rank] if rank < len(key_metas) else {}; doc_details[doc_id] = {"doc": doc_text, "meta": doc_meta}
            if not fused_scores: print("  - No combined results after fusion."); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
            sorted_fused_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
            top_n_fused = sorted_fused_results[:n_results]
            final_ids = [doc_id for doc_id, score in top_n_fused]; final_docs = [doc_details.get(doc_id, {}).get("doc", "N/A") for doc_id in final_ids]; final_metas = [doc_details.get(doc_id, {}).get("meta", {}) for doc_id in final_ids]; final_dists = []
            if top_n_fused:
                scores_only = [score for _, score in top_n_fused]; max_s, min_s = (max(scores_only), min(scores_only)) if scores_only else (1.0, 0.0)
                range_s = max_s - min_s if max_s > min_s else 1.0; final_dists = [max(0.0, 1.0 - ((score - min_s) / range_s)) if range_s > 0 else 0.5 for _, score in top_n_fused]
            return {"ids": [final_ids], "documents": [final_docs], "metadatas": [final_metas], "distances": [final_dists]}
        except Exception as e: st.error(f"Hybrid search encountered an error: {e}"); print(f"Hybrid search error:"); traceback.print_exc(); st.warning("Falling back to semantic search only due to hybrid search error.")
        try: return self.collection.query(query_texts=[query], n_results=n_results, where=where_filter, include=["metadatas", "documents", "distances"])
        except Exception as fb_e: st.error(f"Fallback semantic search also failed: {fb_e}"); print(f"Fallback semantic search error:"); traceback.print_exc(); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        if not chunks: print("No chunks provided to add."); return True
        if not self.collection: st.error("Cannot add documents: Collection is not available."); return False
        try:
            ids, docs, metas = [c["id"] for c in chunks], [c["text"] for c in chunks], [c["metadata"] for c in chunks]
            batch_size, total_chunks = 100, len(chunks); num_batches = math.ceil(total_chunks / batch_size)
            print(f"Adding {total_chunks} chunks to knowledge base...")
            for i in range(num_batches):
                start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, total_chunks)
                batch_ids, batch_docs, batch_metas = ids[start_idx:end_idx], docs[start_idx:end_idx], metas[start_idx:end_idx]
                if not batch_ids: continue
                self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            print(f"Successfully added {total_chunks} chunks.")
            return True
        except Exception as e: st.error(f"Failed to add document chunks: {e}"); print(f"Add documents error:"); traceback.print_exc(); return False

    def query_documents(self, query: str, n_results: int = 3, where_filter: Optional[Dict] = None) -> Optional[Dict[str, List]]:
        if not self.collection: st.error("Cannot query documents: Collection is not available."); return None
        if not query: st.warning("Query cannot be empty."); return None
        try:
            start_time = time.time()
            results = self._hybrid_search(query, n_results=n_results, where_filter=where_filter)
            end_time = time.time(); print(f"Hybrid query execution time: {end_time - start_time:.2f}s")
            if results and isinstance(results.get("documents"), list) and results["documents"] and results["documents"][0]: return results
            elif results: print("Hybrid search completed, but found no matching documents for the query."); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
            else: st.error("Hybrid search returned an unexpected result."); return None
        except Exception as e: st.error(f"An error occurred during document query: {e}"); print(f"Query documents error:"); traceback.print_exc(); return None

    def _check_hallucination(self, query: str, context_docs: list, generated_answer: str) -> Optional[Dict]:
        # Hallucination check logic remains unchanged internally
        if not self.llm: st.warning("Hallucination check skipped: LLM client is not available."); return None
        try:
            # ... (prompt preparation logic unchanged) ...
            prompt = f"""Please act as a fact-checker... [Your prompt remains the same] ... Provide ONLY the JSON object in your response.
            """
            hallucination_model = self.llm_info.get("hallucination_check_model", "gpt-4o-mini")
            print(f"  - Performing hallucination check using {hallucination_model}...")
            response = self.llm.chat.completions.create(model=hallucination_model, messages=[{"role": "system", "content": "You are a meticulous fact-checker. Output ONLY JSON."}, {"role": "user", "content": prompt}], temperature=0.0, response_format={"type": "json_object"})
            analysis_json = response.choices[0].message.content; analysis = json.loads(analysis_json)
            if isinstance(analysis, dict) and "is_grounded" in analysis and "unsupported_statements" in analysis: print(f"  - Hallucination check result: Grounded = {analysis.get('is_grounded')}"); return analysis
            else: st.warning("Hallucination check response was not in the expected JSON format."); print(f"Unexpected hallucination check JSON: {analysis_json}"); return None
        except json.JSONDecodeError: st.warning("Failed to parse hallucination check response as JSON."); print(f"Failed to decode hallucination check JSON: {analysis_json}"); return None
        except Exception as e: st.warning(f"Hallucination check encountered an error: {e}"); print(f"Hallucination check error:"); traceback.print_exc(); return None

    def generate_response(self, query: str, context: Optional[Dict], temperature: float = 0.7, check_for_hallucinations: bool = True) -> Optional[Union[Dict, str]]:
        if not self.llm: return "Error: The AI model (LLM) is currently unavailable."
        if not context or not context.get("documents") or not context["documents"][0]: return "I couldn't find any relevant information in the knowledge base to answer your question."
        raw_docs = context["documents"][0]; raw_metas = context.get("metadatas", [[]])[0]
        try:
            sources_text_list = []
            # <<< CHANGE >>> Store mapping: Citation Label -> {URL, Display Name}
            source_mapping: Dict[str, Dict[str, str]] = {}
            if len(raw_docs) == len(raw_metas):
                for i, (doc, meta) in enumerate(zip(raw_docs, raw_metas)):
                    source_url = meta.get('source', f'Unknown Source URL {i+1}') if meta else f'Unknown Source URL {i+1}'
                    display_name = meta.get('display_name', f'Unknown Source Name {i+1}') if meta else f'Unknown Source Name {i+1}'
                    citation_label = f"Source {i+1}"
                    sources_text_list.append(f"[{citation_label}]\n{doc}")
                    source_mapping[citation_label] = {"url": source_url, "display_name": display_name} # Store both
            else:
                st.warning("Mismatch between number of documents and metadatas. Using generic source names.")
                print("WARNING: Mismatch doc/meta counts in generate_response context.")
                for i, doc in enumerate(raw_docs):
                     citation_label = f"Source {i+1}"
                     sources_text_list.append(f"[{citation_label}]\n{doc}")
                     source_mapping[citation_label] = {"url": f"Unknown Source URL {i+1}", "display_name": f"Unknown Source Name {i+1}"}

            formatted_context = "\n\n---\n\n".join(sources_text_list)
            # Prompt remains largely the same, focusing on citation labels
            prompt = f"""You are an assistant tasked with answering a user's question based *only* on the provided text excerpts... [rest of prompt unchanged] ... Generate the JSON response now based *only* on the provided excerpts:"""
            qa_model = self.llm_info.get("qa_model", "gpt-4o-mini"); print(f"Generating response using {qa_model} with temp={temperature}...")
            response = self.llm.chat.completions.create(model=qa_model, messages=[{"role": "system", "content": "You are an expert assistant analyzing text excerpts..."}, {"role": "user", "content": prompt}], temperature=temperature, response_format={"type": "json_object"})
            raw_response_content = response.choices[0].message.content.strip()
            try:
                structured_answer = json.loads(raw_response_content)
                required_keys = ["direct_answer", "detailed_explanation", "key_points"]
                if not isinstance(structured_answer, dict) or not all(k in structured_answer for k in required_keys):
                    st.warning("LLM response missing required keys."); print(f"Warning: LLM JSON response missing keys. Raw response: {raw_response_content}")
                    return f"LLM Response (unexpected format):\n```json\n{raw_response_content}\n```"
                structured_answer["source_mapping"] = source_mapping # Add the {url, display_name} mapping
                structured_answer["hallucination_warning"] = None
                if check_for_hallucinations: # Hallucination check logic unchanged
                    text_to_check = structured_answer.get("detailed_explanation", "")
                    if text_to_check:
                        hallucination_result = self._check_hallucination(query, raw_docs, text_to_check)
                        if hallucination_result and not hallucination_result.get("is_grounded", True): structured_answer["hallucination_warning"] = "**⚠️ Potential Hallucination Warning**"
                return structured_answer
            except json.JSONDecodeError as json_e: st.error(f"Failed to parse AI's response as JSON: {json_e}"); print(f"JSONDecodeError. Raw content:\n{raw_response_content}"); return f"Error: Couldn't understand AI's structured answer. Raw response:\n```\n{raw_response_content}\n```"
            except Exception as parse_e: st.error(f"Error processing AI's response: {parse_e}"); print(f"Error processing LLM response:"); traceback.print_exc(); return f"Error processing AI response: {parse_e}"
        except Exception as e: st.error(f"Unexpected error during response generation: {e}"); print(f"Generate response error:"); traceback.print_exc(); return f"An error occurred while generating the response: {e}"

    def get_embedding_info(self) -> Dict[str, Any]:
        if not self.emb_info: return {}
        return {"name": self.emb_info.get("name", "Unknown"), "dimensions": self.emb_info.get("dimensions", "N/A"), "model_provider": self.embedding_model_provider, "model_name": self.emb_info.get("model_name", "N/A")}

    # <<< CHANGE >>> Return unique display names as 'sources'
    def get_collection_stats(self) -> Dict[str, Any]:
        """Retrieves statistics including unique display names."""
        if not self.collection: return {"count": 0, "sources": [], "source_note": ""}
        try:
            count = self.collection.count()
            # Use the internal mapping if already populated, otherwise fetch
            if self.display_name_to_url: # Check if mapping exists
                 display_names = sorted(list(self.display_name_to_url.keys()))
                 source_note = "" # Assuming mapping is complete if it exists
                 print(f"Getting stats from existing mappings: {len(display_names)} sources.")
            else:
                 # Fallback: Fetch metadata if mapping isn't populated yet (e.g., initial load failed partially)
                 print("Mappings not populated, fetching metadata for stats...")
                 limit = 10000
                 items = self.collection.get(limit=limit, include=['metadatas'])
                 display_names_set = set()
                 if items and items.get("metadatas"):
                      for m in items["metadatas"]:
                           if m and "display_name" in m:
                                display_names_set.add(m["display_name"])
                 display_names = sorted(list(display_names_set))
                 source_note = f"(from first {limit} chunks)" if count > limit and len(display_names_set) < count else "" # Rough estimate

            return {"count": count, "sources": display_names, "source_note": source_note} # sources key now holds display names
        except Exception as e:
            st.error(f"Error getting collection stats: {e}")
            print(f"Get collection stats error:"); traceback.print_exc()
            return {"count": 0, "sources": [], "source_note": "(Error retrieving stats)"}

    # <<< CHANGE >>> Delete by URL, but could be triggered by display name lookup if needed
    # Keeping it simple: delete by URL for now. If UI needs delete, it needs to map name -> URL first.
    def delete_document_by_source(self, source_url: str) -> bool:
         # Deletion logic remains the same, operating on the unique 'source' URL
         if not source_url: st.warning("No source URL provided for deletion."); return False
         if not self.collection: st.error("Cannot delete document: Collection is not available."); return False
         try:
             st.info(f"Finding chunks associated with source URL: '{source_url}'...")
             results = self.collection.get(where={"source": source_url}, include=[]) # Filter by URL
             ids_to_delete = results.get("ids")
             if not ids_to_delete: st.warning(f"No document chunks found with source URL '{source_url}'."); return False
             st.info(f"Found {len(ids_to_delete)} chunks. Deleting content for URL '{source_url}'...")
             self.collection.delete(ids=ids_to_delete)
             print(f"Deleted {len(ids_to_delete)} chunks for source URL '{source_url}'.")
             print("Updating keyword index and mappings after deletion...")
             time.sleep(0.5)
             self._update_keyword_search_index_from_db() # This will also rebuild mappings
             return True
         except Exception as e: st.error(f"An error occurred while deleting content for URL '{source_url}': {e}"); print(f"Delete document by source URL error:"); traceback.print_exc(); return False

    def load_knowledge_from_urls(self, force_reload: bool = False):
        url_name_pairs = self._read_url_config() # Now returns [(name, url)]
        if not url_name_pairs: st.warning("No valid URL entries found in the configuration file."); return
        if not self.collection: st.error("Cannot load knowledge: Collection is not available."); return

        # Get existing SOURCE URLs from the DB to check for reprocessing
        existing_source_urls = set()
        try:
             # Fetch only metadata, potentially limited if huge collection
             limit = 20000
             meta_data = self.collection.get(limit=limit, include=['metadatas'])
             if meta_data and meta_data.get("metadatas"):
                  existing_source_urls = {m['source'] for m in meta_data['metadatas'] if m and 'source' in m}
             print(f"Found {len(existing_source_urls)} existing source URLs in DB (checked up to {limit} items).")
        except Exception as e:
             print(f"Warning: Could not efficiently fetch existing source URLs: {e}. May reprocess existing URLs.")
             force_reload = True # Force reload if we can't check existing

        urls_to_process_dict = {} # Use dict to preserve name association {url: name}
        if force_reload:
            print("Force reload requested or check failed. Processing all configured URLs.")
            urls_to_process_dict = {url: name for name, url in url_name_pairs}
        else:
            for name, url in url_name_pairs:
                if url not in existing_source_urls:
                    urls_to_process_dict[url] = name
                else:
                    print(f"Skipping already processed URL: {url} ({name})")

        if not urls_to_process_dict:
            print("No new URLs to process.")
            if self.bm25 is None and self.collection.count() > 0:
                 print("Knowledge already loaded, ensuring BM25 index and mappings exist...")
                 self._update_keyword_search_index_from_db() # Will update mappings too
            return

        st.info(f"Processing {len(urls_to_process_dict)} new URL source(s)...")
        processor = TheURLProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        urls_processed_count = 0; urls_failed_count = 0
        status_placeholder = st.empty()

        # Sort items for consistent processing order (optional)
        sorted_urls_to_process = sorted(urls_to_process_dict.items())

        for i, (url, display_name) in enumerate(sorted_urls_to_process):
            status_placeholder.info(f"⚙️ Processing {i+1}/{len(urls_to_process_dict)}: '{display_name}'")
            try:
                extracted_text = processor.fetch_and_extract_text(url)
                if extracted_text:
                    # <<< CHANGE >>> Pass display_name to create_chunks
                    chunks = processor.create_chunks(extracted_text, source_url=url, display_name=display_name)
                    if chunks:
                        success = self.add_documents(chunks)
                        if success: urls_processed_count += 1
                        else: urls_failed_count += 1; st.warning(f"Failed to add chunks for: {display_name}", icon="⚠️")
                    else: urls_failed_count += 1; st.warning(f"Failed to create chunks for: {display_name}", icon="🧩")
                else: urls_failed_count += 1
            except Exception as e: urls_failed_count += 1; st.error(f"Error processing {display_name}: {e}"); print(f"Error processing {url}: {e}"); traceback.print_exc()

        status_placeholder.empty()

        if urls_processed_count > 0:
             print("Updating keyword index and mappings after processing new URLs...")
             self._update_keyword_search_index_from_db() # This updates BM25 and URL mappings

        # Final status message
        if urls_failed_count == 0 and urls_processed_count > 0: st.success(f"Successfully processed {urls_processed_count} new source(s). Knowledge base updated.")
        elif urls_processed_count > 0: st.warning(f"Processed {urls_processed_count} source(s), but failed to process {urls_failed_count}. Check logs/console.")
        elif urls_failed_count > 0: st.error(f"Failed to process {urls_failed_count} new source(s). Check logs/console.")
        else: print("No new URLs were processed.") # Should have been caught earlier


    # <<< CHANGE >>> Read 'Name | URL' format
    def _read_url_config(self) -> List[Tuple[str, str]]:
        """Reads 'Display Name | URL' pairs from the config file."""
        url_pairs = []
        malformed_lines = 0
        try:
            with open(URL_CONFIG_FILE, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    if '|' not in line:
                         print(f"Warning: Malformed line {i+1} in '{URL_CONFIG_FILE}' (missing '|'): {line}")
                         malformed_lines += 1
                         continue
                    parts = line.split('|', 1) # Split only on the first pipe
                    if len(parts) == 2:
                        name = parts[0].strip()
                        url = parts[1].strip()
                        if name and url: url_pairs.append((name, url))
                        else: print(f"Warning: Malformed line {i+1} in '{URL_CONFIG_FILE}' (empty name or URL): {line}"); malformed_lines += 1
                    else: # Should not happen with split('|', 1) but safety check
                        print(f"Warning: Malformed line {i+1} in '{URL_CONFIG_FILE}': {line}"); malformed_lines += 1

            print(f"Read {len(url_pairs)} URL pairs from '{URL_CONFIG_FILE}'.")
            if malformed_lines > 0:
                 st.warning(f"{malformed_lines} lines in '{URL_CONFIG_FILE}' seem malformed (check format: Name | URL).", icon="📄")

        except FileNotFoundError: st.error(f"URL configuration file not found: '{URL_CONFIG_FILE}'."); print(f"ERROR: URL configuration file not found: '{URL_CONFIG_FILE}'")
        except Exception as e: st.error(f"Error reading URL configuration file: {e}"); print(f"Error reading URL config file: {e}")
        return url_pairs

# --- Streamlit UI (Adapted for Display Names) ---
def main():
    st.set_page_config(page_title="SmartBot URL Q&A", page_icon="🌐", layout="wide")
    st.title("🌐 SmartBot URL Q&A")
    st.caption("Ask questions and get answers grounded in the knowledge from pre-configured web pages.")

    if "rag_system" not in st.session_state: st.session_state.rag_system = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "knowledge_loaded" not in st.session_state: st.session_state.knowledge_loaded = False

    try:
        if st.session_state.rag_system is None:
            print("Initializing RAG System...")
            model_selector_init = TheModelSelector()
            llm_provider, embedding_provider = model_selector_init.get_models()
            st.session_state.rag_system = TheRAGSystem(embedding_model_provider=embedding_provider, llm_provider=llm_provider)
            print("RAG System Initialized.")
        rag_system: TheRAGSystem = st.session_state.rag_system

        if not st.session_state.knowledge_loaded:
             print("Knowledge not loaded yet. Attempting to load from URLs...")
             with st.spinner("Connecting to Knowledge Base and loading URL content..."):
                  force = False # Set True to always re-process URLs on startup
                  rag_system.load_knowledge_from_urls(force_reload=force)
                  st.session_state.knowledge_loaded = True
             print("Knowledge loading process complete.")

        # --- Sidebar Information ---
        with st.sidebar:
            st.header("📊 Knowledge Base Status")
            stats = rag_system.get_collection_stats()
            st.metric("Total Indexed Chunks", stats.get("count", 0))
            # <<< CHANGE >>> Show display names
            sources_list = stats.get("sources", []) # Now contains display names
            st.metric("Indexed Sources", len(sources_list))
            st.write("**Indexed Sources:**")
            if sources_list:
                max_sources_display = 15
                for i, name in enumerate(sources_list):
                    if i < max_sources_display: st.caption(f"- {name}")
                    elif i == max_sources_display: st.caption(f"- ... and {len(sources_list) - max_sources_display} more"); break
            else: st.caption("No sources indexed yet.")
            if stats.get("source_note"): st.caption(stats["source_note"])

            st.divider()
            if st.button("🔄 Refresh Knowledge Base", help="Re-check URL list and process any new URLs."):
                 with st.spinner("Refreshing knowledge from URLs..."):
                      rag_system.load_knowledge_from_urls(force_reload=False) # Checks for new URLs based on URL
                 st.success("Knowledge refresh check complete.")
                 time.sleep(1); st.rerun()

            st.divider()
            st.header("⚙️ Configuration")
            model_selector_disp = TheModelSelector(); llm_disp_info = model_selector_disp.get_llm_info(); emb_disp_info = rag_system.get_embedding_info()
            st.info(f"**LLM:** {llm_disp_info['qa_model']} (for Q&A)\n"
                    f"**Embeddings:** {emb_disp_info['name']} ({emb_disp_info['model_provider']}, Dim: {emb_disp_info['dimensions']})\n"
                    f"**Retrieval:** Hybrid Search (RRF)")
            st.divider()

    except Exception as e: st.error(f"Fatal error during RAG system initialization or loading: {e}"); print("Fatal RAG Initialization/Loading Error:"); traceback.print_exc(); st.stop()

    # --- Main Chat Interface ---
    st.header("Ask Questions About the Indexed Web Pages")
    current_stats = rag_system.get_collection_stats() # Contains display names in 'sources'

    if current_stats["count"] == 0 and st.session_state.knowledge_loaded:
        st.warning("The knowledge base is currently empty or failed to load. Check `knowledge_urls.txt` and logs.")
    elif not st.session_state.knowledge_loaded: st.info("Knowledge base is loading...")
    else:
        col_context, col_temp = st.columns([3, 1])
        with col_context:
            # <<< CHANGE >>> Use display names for options
            available_sources = ["All Sources"] + current_stats.get("sources", []) # Get display names
            selected_context_name = st.selectbox( # Variable name clarifies it's the display name
                "Limit context to specific source (optional):",
                options=available_sources,
                key="chat_context_select",
                help="Choose 'All Sources' or select a specific source name."
            )
        with col_temp:
            temperature = st.slider("LLM Temperature (Creativity)", 0.0, 1.0, 0.5, 0.05, key="chat_temp_slider")

        # <<< CHANGE >>> Filter by display_name if a specific source is selected
        query_filter = None
        if selected_context_name != "All Sources":
             query_filter = {"display_name": selected_context_name}
             st.caption(f"ℹ️ Answers will be based primarily on content from: **{selected_context_name}**")

        st.markdown("---")
        st.subheader("Conversation")
        if not st.session_state.chat_history: st.caption("No questions asked yet.")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        user_query = st.chat_input("Enter your question here...")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.markdown(user_query)
            with st.chat_message("assistant"):
                placeholder = st.empty(); placeholder.markdown("Thinking... 🤔")
                formatted_response_content = ""
                try:
                    start_time = time.time()
                    context = rag_system.query_documents(query=user_query, n_results=5, where_filter=query_filter)
                    response_object = rag_system.generate_response(query=user_query, context=context, temperature=temperature, check_for_hallucinations=False)
                    end_time = time.time(); print(f"Chat query processing time: {end_time - start_time:.2f}s")

                    if isinstance(response_object, dict):
                        display_parts = []
                        # ... (building direct_answer, explanation, key_points unchanged) ...
                        if response_object.get("hallucination_warning"): display_parts.append(response_object["hallucination_warning"])
                        if response_object.get("direct_answer"): display_parts.append(f"**Answer:** {response_object['direct_answer']}")
                        if response_object.get("detailed_explanation"): prefix = "\n---\n" if display_parts else ""; display_parts.append(f"{prefix}**Explanation:**\n{response_object['detailed_explanation']}")
                        if response_object.get("key_points"): points_markdown = "\n".join([f"- {point}" for point in response_object["key_points"]]); prefix = "\n---\n" if display_parts else ""; display_parts.append(f"{prefix}**Key Points:**\n{points_markdown}")
                        formatted_response_content = "\n\n".join(display_parts)

                        # <<< CHANGE >>> Source Expander uses display name and URL
                        sources = context.get("documents", [[]])[0] if context else []
                        metadatas = context.get("metadatas", [[]])[0] if context else []
                        source_mapping = response_object.get("source_mapping", {}) # Label -> {url, display_name}

                        if sources and metadatas and source_mapping:
                             with st.expander("View retrieved source document snippets", expanded=False):
                                # Need a map from URL back to citation label for display consistency?
                                # Or just iterate through context and use metadata directly? Let's use context directly.
                                for idx, (doc_text, meta) in enumerate(zip(sources, metadatas)):
                                    source_url = meta.get("source", "Unknown URL")
                                    display_name = meta.get("display_name", "Unknown Source")
                                    # Attempt to find the matching citation label from the response mapping (more complex)
                                    # Easier: Just display name and URL from the retrieved context metadata.
                                    st.markdown(f"**{idx+1}. {display_name}** (`{source_url}`) (Chars: {meta.get('start', '?')}-{meta.get('end', '?')})")
                                    display_doc = doc_text[:600] + "..." if len(doc_text) > 600 else doc_text
                                    st.text_area(f"Source Snippet {idx}", display_doc, height=120, disabled=True, label_visibility="collapsed", key=f"chat_src_{idx}_{user_query[:10]}" )
                                st.caption("Note: These are the raw text snippets provided to the AI. Citations like [Source X] in the answer above refer to these snippets based on the AI's synthesis.")

                    elif isinstance(response_object, str): formatted_response_content = response_object
                    else: formatted_response_content = "Sorry, I received an unexpected response format."; print(f"Unexpected response type: {type(response_object)}")
                    placeholder.markdown(formatted_response_content)
                except Exception as chat_e: formatted_response_content = f"An error occurred: {chat_e}"; placeholder.error(formatted_response_content); print(f"Chat processing error:"); traceback.print_exc()
            st.session_state.chat_history.append({"role": "assistant", "content": formatted_response_content})

        st.markdown("---")
        if st.button("Clear Chat History", key="clear_chat_btn"): st.session_state.chat_history = []; st.rerun()

if __name__ == "__main__":
    main()