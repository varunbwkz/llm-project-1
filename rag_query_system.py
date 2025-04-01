# import streamlit as st
# import chromadb
# from chromadb.utils import embedding_functions
# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# import PyPDF2
# import docx
# import uuid
# import io
# import time
# import concurrent.futures
# import math
# import json
# import re
# from typing import Union, Optional, Dict, List, Any, Tuple
# import numpy as np
# from rank_bm25 import BM25Okapi

# # --- Load Environment Variables ---
# load_dotenv()

# # --- Constants ---
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# MAX_WORKERS = 4  # For parallel processing

# # --- Model Selection ---
# class TheModelSelector:
#     """Handles AI model selection. Currently focused on OpenAI."""
#     def __init__(self):
#         self.llm_model = "openai"
#         self.embedding_model = "openai"
#         self.embedding_models = {
#             "openai": {
#                 "name": "OpenAI Embeddings",
#                 "dimensions": 1536,
#                 "model_name": "text-embedding-3-small",
#             }
#         }

#     def get_models(self) -> Tuple[str, str]:
#         """Returns the selected LLM and embedding model names."""
#         return self.llm_model, self.embedding_model

#     def get_embedding_info(self, model_key: str) -> Optional[Dict[str, Any]]:
#          """Gets details for a specific embedding model."""
#          return self.embedding_models.get(model_key)

# # --- Document Processing (Handles PDF, DOCX, TXT) ---
# class TheDocProcessor:
#     """Handles reading documents (PDF, DOCX, TXT) and splitting them into manageable chunks."""

#     def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         if self.chunk_overlap >= self.chunk_size:
#              st.warning(f"Chunk overlap ({self.chunk_overlap}) >= chunk size ({self.chunk_size}). Setting overlap to {self.chunk_size // 5}.")
#              self.chunk_overlap = self.chunk_size // 5

#     def read_document(self, uploaded_file: io.BytesIO, file_name: str) -> str:
#         """Reads text content from PDF, DOCX, or TXT files."""
#         text = ""
#         try:
#             uploaded_file.seek(0) # Reset file pointer
#             file_extension = os.path.splitext(file_name)[1].lower()
#             st.info(f"Reading {file_extension.upper()} file: '{file_name}'...")

#             if file_extension == ".pdf":
#                 reader = PyPDF2.PdfReader(uploaded_file)
#                 total_pages = len(reader.pages)
#                 progress_bar = None
#                 if total_pages > 10:
#                     progress_text = f"Reading PDF '{file_name}' ({total_pages} pages)..."
#                     try: progress_bar = st.progress(0, text=progress_text)
#                     except: st.info(progress_text); progress_bar = None
#                 for i, page in enumerate(reader.pages):
#                     try:
#                         page_text = page.extract_text()
#                         if page_text: text += page_text + "\n"
#                     except Exception as e: st.warning(f"Could not extract text from PDF page {i+1} of '{file_name}': {e}")
#                     if progress_bar:
#                         try: progress_bar.progress((i + 1) / total_pages, text=progress_text)
#                         except: progress_bar = None
#                 if progress_bar:
#                     try: progress_bar.empty()
#                     except: pass

#             elif file_extension == ".docx":
#                 document = docx.Document(uploaded_file)
#                 full_text = [para.text for para in document.paragraphs]
#                 text = "\n".join(full_text)

#             elif file_extension == ".txt":
#                 content_bytes = uploaded_file.read()
#                 try: text = content_bytes.decode('utf-8')
#                 except UnicodeDecodeError:
#                     st.warning("UTF-8 decoding failed, trying latin-1...")
#                     try: text = content_bytes.decode('latin-1')
#                     except Exception as decode_err:
#                         st.error(f"Failed to decode TXT file '{file_name}': {decode_err}")
#                         return ""

#             else:
#                 st.error(f"Unsupported file type: '{file_extension}'. Please upload PDF, DOCX, or TXT.")
#                 return ""

#             # Basic text cleaning
#             text = re.sub(r'\s+', ' ', text).strip()
#             if not text: st.warning(f"No text content extracted from '{file_name}'.")
#             return text

#         except PyPDF2.errors.PdfReadError:
#             st.error(f"Could not read PDF '{file_name}'. Corrupted or password-protected?")
#             return ""
#         except Exception as e:
#             st.error(f"Error reading '{file_name}': {e}")
#             print(f"Error reading {file_name}: {e}")
#             return ""

#     def process_doc_chunk(self, chunk_data: Tuple[int, int, str, str]) -> Dict[str, Any]:
#         """Processes one piece of the document text."""
#         start, end, text, doc_name = chunk_data
#         effective_start = max(0, start - self.chunk_overlap if start > 0 else 0)
#         chunk_text = text[effective_start:end]
#         final_end = end
#         if end < len(text):
#             possible_ends = [m.start() + 1 for m in re.finditer(r'[.!?]\s', chunk_text)]
#             if possible_ends:
#                 ideal_end = end - effective_start; best_end = -1; min_diff = float('inf')
#                 for p_end in possible_ends:
#                     if abs(p_end - ideal_end) < self.chunk_size * 0.3:
#                         diff = abs(p_end - ideal_end)
#                         if diff < min_diff: min_diff = diff; best_end = p_end
#                 if best_end != -1: final_end = effective_start + best_end
#             chunk_text = text[effective_start:final_end]
#         else:
#              final_end = len(text)
#              chunk_text = text[effective_start:final_end]
#         chunk_text = chunk_text.strip()
#         return {"id": str(uuid.uuid4()), "text": chunk_text, "metadata": {"source": doc_name, "start": effective_start, "end": final_end}, "end_pos": final_end}


#     def create_chunks(self, text: str, file_obj: io.BytesIO) -> List[Dict[str, Any]]:
#         """Splits document text into chunks."""
#         if not text: return []
#         chunks = []; start = 0; doc_len = len(text); file_name = file_obj.name

#         # Determine processing method based on size
#         use_parallel = doc_len > 100000 # Threshold for parallel processing

#         if not use_parallel:
#             print(f"Using sequential chunking for '{file_name}'...")
#             while start < doc_len:
#                 end = min(start + self.chunk_size, doc_len)
#                 chunk_data = (start, end, text, file_name)
#                 processed = self.process_doc_chunk(chunk_data)
#                 if processed["text"]: chunks.append({"id": processed["id"], "text": processed["text"], "metadata": processed["metadata"]})
#                 start = processed["end_pos"]
#                 if start <= (max(0, end - self.chunk_size - self.chunk_overlap)): start = max(start + 1, end - self.chunk_overlap + 1) # Prevent stall
#         else:
#             st.info(f"Using parallel chunking for '{file_name}'...")
#             tasks = []
#             while start < doc_len:
#                 end = min(start + self.chunk_size, doc_len)
#                 tasks.append((start, end, text, file_name))
#                 start = end
#             total = len(tasks); prog = None; prog_txt=f"Chunking... 0/{total}"
#             try: prog = st.progress(0, text=prog_txt)
#             except: st.info(prog_txt); prog = None
#             results = []
#             with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                 futures = {executor.submit(self.process_doc_chunk, task): task for task in tasks}
#                 count = 0
#                 for future in concurrent.futures.as_completed(futures):
#                     try:
#                         res = future.result(); results.append(res); count += 1
#                         if prog:
#                            try: prog.progress(count/total, text=f"Chunking... {count}/{total}")
#                            except: prog = None
#                     except Exception as exc: st.warning(f"Chunk failed: {exc}")
#             if prog:
#                  try: prog.empty()
#                  except: pass
#             results.sort(key=lambda r: r['metadata']['start'])
#             chunks = [{"id": r["id"], "text": r["text"], "metadata": r["metadata"]} for r in results if r["text"]]

#         st.write(f"Generated {len(chunks)} chunks for '{file_name}'.")
#         return chunks


# class TheRAGSystem:
#     """Stores documents, performs hybrid search (semantic + keyword), and generates answers."""

#     def __init__(self, embedding_model="openai", llm_model="openai"):
#         self.embedding_model_name = embedding_model; self.llm_model_name = llm_model
#         self.db_path = "./chroma_db"; self.collection_name = f"documents_{self.embedding_model_name}"
#         # Setup DB, Embedding Fn, LLM (Error handling included)
#         try: self.db = chromadb.PersistentClient(path=self.db_path)
#         except Exception as e: st.error(f"DB init error: {e}"); st.stop()
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key: st.error("OPENAI_API_KEY missing!"); st.stop()
#         model_selector = TheModelSelector(); emb_info = model_selector.get_embedding_info(self.embedding_model_name)
#         if not emb_info: st.error(f"Embedding info missing: '{self.embedding_model_name}'"); st.stop()
#         try: self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=self.openai_api_key, model_name=emb_info['model_name'])
#         except Exception as e: st.error(f"Embedding fn error: {e}"); st.stop()
#         try: self.llm = OpenAI(api_key=self.openai_api_key)
#         except Exception as e: st.error(f"LLM client error: {e}"); self.llm = None
#         self.collection = self._setup_collection()
#         # BM25 Index Components
#         self.corpus: List[str] = []; self.doc_ids: List[str] = []; self.doc_metadatas: List[Dict] = []; self.bm25: Optional[BM25Okapi] = None
#         self._update_keyword_search_index_from_db()

#     def _setup_collection(self) -> chromadb.Collection:
#         try:
#             collection = self.db.get_or_create_collection(name=self.collection_name, embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"})
#             print(f"Using collection: '{self.collection_name}'"); return collection
#         except Exception as e: st.error(f"Collection setup error: {e}"); raise e

#     def _update_keyword_search_index_from_db(self):
#         print("Updating keyword index...")
#         if not self.collection: print("Warn: Collection not ready."); return
#         try:
#             all_data = self.collection.get(include=["metadatas", "documents"])
#             if not all_data or not all_data.get("ids"):
#                 print("Keyword index empty."); self.corpus=[]; self.doc_ids=[]; self.doc_metadatas=[]; self.bm25=None; return
#             self.doc_ids=all_data["ids"]; self.corpus=all_data["documents"]; self.doc_metadatas=all_data["metadatas"]
#             if not self.corpus: print("Warn: No doc text found."); self.bm25=None; return
#             print(f"Building BM25 for {len(self.corpus)} docs...")
#             tokenized_corpus = [doc.lower().split() for doc in self.corpus] # Simple tokenization
#             self.bm25 = BM25Okapi(tokenized_corpus)
#             print("Keyword index updated.")
#         except Exception as e: st.error(f"Keyword index update error: {e}"); print(f"Keyword index update error: {e}"); self.bm25=None

#     def _keyword_search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> Dict[str, List]:
#         if not self.bm25 or not self.corpus: return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
#         tokens = query.lower().split(); scores = self.bm25.get_scores(tokens)
#         res_scores = []
#         for i, score in enumerate(scores):
#             if i >= len(self.doc_metadatas) or i >= len(self.doc_ids): continue
#             meta = self.doc_metadatas[i]
#             if where_filter and not all(str(meta.get(k)) == str(v) for k, v in where_filter.items()): continue
#             res_scores.append((i, self.doc_ids[i], score, meta))
#         res_scores.sort(key=lambda x: x[2], reverse=True); top_res = res_scores[:n_results]
#         ids = [r[1] for r in top_res]; docs = [self.corpus[r[0]] for r in top_res]; metas = [r[3] for r in top_res]
#         dists = []
#         if top_res:
#             scores_only = [s[2] for s in top_res]; max_s=max(scores_only) if scores_only else 1.0; min_s=min(scores_only) if scores_only else 0.0; range_s = max_s-min_s if max_s>min_s else 1.0
#             dists = [max(0.0, 1.0 - (s[2]-min_s)/range_s) if range_s > 0 else 0.5 for s in top_res]
#         return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}


#     def _hybrid_search(self, query: str, n_results: int = 5, where_filter: Optional[Dict] = None) -> Dict[str, List]:
#         print("Performing Hybrid Search (Semantic + Keyword)...")
#         if not self.collection: st.error("Collection error."); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
#         try:
#             sem_n=max(n_results*2, 10); key_n=max(n_results*2, 10)
#             print(f"Hybrid: Sem(n={sem_n}), Key(n={key_n})")
#             sem_res = self.collection.query(query_texts=[query], n_results=sem_n, where=where_filter, include=["metadatas","documents","distances"])
#             key_res = self._keyword_search(query, n_results=key_n, where_filter=where_filter)
#             comb_res: Dict[str, Dict[str, Any]] = {}
#             if sem_res and sem_res.get("ids") and sem_res["ids"][0]:
#                 print(f"  Processing {len(sem_res['ids'][0])} semantic.")
#                 for i, id_ in enumerate(sem_res["ids"][0]):
#                     if id_ and i<len(sem_res["documents"][0]) and i<len(sem_res["metadatas"][0]) and i<len(sem_res["distances"][0]):
#                         comb_res[id_] = {"doc": sem_res["documents"][0][i], "meta": sem_res["metadatas"][0][i], "score": 1.0 - sem_res["distances"][0][i]}
#             if key_res and key_res.get("ids") and key_res["ids"][0]:
#                 print(f"  Processing {len(key_res['ids'][0])} keyword.")
#                 if key_res.get("distances") and key_res["distances"][0]:
#                     kw_dists = key_res["distances"][0]
#                     for i, id_ in enumerate(key_res["ids"][0]):
#                         if id_ and i<len(key_res["documents"][0]) and i<len(key_res["metadatas"][0]) and i<len(kw_dists):
#                             kw_score = 1.0 - kw_dists[i]
#                             if id_ in comb_res: comb_res[id_]["score"] = (comb_res[id_]["score"] + kw_score) / 2
#                             else: comb_res[id_] = {"doc": key_res["documents"][0][i], "meta": key_res["metadatas"][0][i], "score": kw_score}
#             if not comb_res: print("  No combined results."); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
#             sorted_res = sorted(comb_res.items(), key=lambda item: item[1]["score"], reverse=True)
#             print(f"  Combined unique sorted: {len(sorted_res)}")
#             final_ids = [r[0] for r in sorted_res[:n_results]]; final_docs = [r[1]["doc"] for r in sorted_res[:n_results]]
#             final_metas = [r[1]["meta"] for r in sorted_res[:n_results]]; final_dists = [1.0 - r[1]["score"] for r in sorted_res[:n_results]]
#             print(f"Hybrid returning top {len(final_docs)} combined.")
#             return {"ids": [final_ids], "documents": [final_docs], "metadatas": [final_metas], "distances": [final_dists]}
#         except Exception as e: st.error(f"Hybrid search error: {e}"); print(f"Hybrid error: {e}"); st.warning("Fallback: semantic search.");
#         try: return self.collection.query(query_texts=[query], n_results=n_results, where=where_filter, include=["metadatas","documents","distances"])
#         except Exception as fb_e: st.error(f"Fallback failed: {fb_e}"); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}


#     def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
#         """Adds chunks and updates index."""
#         if not chunks: return True
#         try:
#             # ... [Batching and adding logic - same as before] ...
#             if not self.collection: self.collection = self._setup_collection()
#             batch_size = 100; total_chunks = len(chunks); num_batches = math.ceil(total_chunks / batch_size)
#             ids = [c["id"] for c in chunks]; docs = [c["text"] for c in chunks]; metas = [c["metadata"] for c in chunks]
#             prog = None
#             if total_chunks > batch_size:
#                 try: prog = st.progress(0, text=f"Adding {total_chunks} chunks...")
#                 except: st.info(f"Adding {total_chunks} chunks...")
#             for i in range(num_batches):
#                 s, e = i*batch_size, min((i+1)*batch_size, total_chunks)
#                 b_ids, b_docs, b_metas = ids[s:e], docs[s:e], metas[s:e]
#                 if not b_ids: continue
#                 self.collection.add(ids=b_ids, documents=b_docs, metadatas=b_metas)
#                 if prog:
#                     try: prog.progress((i+1)/num_batches, text=f"Adding batch {i+1}/{num_batches}...")
#                     except: prog=None
#             if prog:
#                 try: prog.empty()
#                 except: pass
#             print("Add docs done. Updating index..."); self._update_keyword_search_index_from_db()
#             return True
#         except Exception as e: st.error(f"Add docs error: {e}"); print(f"Add docs error: {e}"); return False


#     def query_documents(self, query: str, n_results: int = 3, where_filter: Optional[Dict] = None) -> Optional[Dict[str, List]]:
#         """Searches docs using HYBRID SEARCH."""
#         if not self.collection: st.error("Collection error."); return None
#         if not query: st.warning("Empty query."); return None
#         try:
#             results = self._hybrid_search(query, n_results=n_results, where_filter=where_filter)
#             if results and "documents" in results and results["documents"] and results["documents"][0]: return results
#             elif results: print("Hybrid OK, no docs."); return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
#             else: st.error("Hybrid invalid result."); return None
#         except Exception as e: st.error(f"Query docs error: {e}"); print(f"Query docs error: {e}"); return None


#     def _check_hallucination(self, query: str, context_docs: list, generated_answer: str) -> Optional[Dict]:
#         """Checks answer grounding."""
#         if not self.llm: return None
#         try:
#             context_str = "\n\n---\n\n".join(f"Doc {i+1}:\n{doc}" for i, doc in enumerate(context_docs))
#             max_len = 15000; context_str = context_str[:max_len] + ("..." if len(context_str)>max_len else "")
#             prompt = f"""Fact Check: Is the 'Generated Answer' fully supported ONLY by the 'Context Documents'?
#             Query: {query}
#             Context Documents:\n{context_str}\n---
#             Generated Answer:\n{generated_answer}\n---
#             Output ONLY JSON: {{"is_grounded": bool, "unsupported_statements": [{{"statement": "...", "reason": "..."}}]}}"""
#             resp = self.llm.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system", "content":"Fact-checker. Output ONLY JSON."}, {"role":"user", "content":prompt}], temperature=0.0, response_format={"type":"json_object"})
#             analysis = json.loads(resp.choices[0].message.content)
#             return analysis if isinstance(analysis, dict) and "is_grounded" in analysis else None
#         except Exception as e: st.warning(f"Hallucination check error: {e}"); return None


#     def generate_response(self, query: str, context: Dict, temperature: float = 0.7, check_for_hallucinations: bool = True) -> Optional[Union[Dict, str]]: # Return type updated
#         """Generates structured answer (Dict) or fallback string from context."""
#         if not self.llm:
#             return "Error: LLM unavailable."
#         if not context or not context.get("documents") or not context["documents"][0]:
#             return "No relevant context documents found to answer the question." # More specific message

#         raw_docs = context["documents"][0]
#         raw_metas = context.get("metadatas", [[]])[0]

#         try:
#             sources_text = []
#             source_mapping = {} # To map simple index to actual source name for citation later
#             if len(raw_docs) == len(raw_metas):
#                 for i, (doc, meta) in enumerate(zip(raw_docs, raw_metas)):
#                     source_name = meta.get('source', f'Document {i+1}') if meta else f'Document {i+1}'
#                     source_id = f"Source {i+1}"
#                     sources_text.append(f"[{source_id}] {doc}")
#                     source_mapping[source_id] = source_name # Map "Source 1" to "filename.pdf"
#             else:
#                 for i, doc in enumerate(raw_docs):
#                      source_id = f"Source {i+1}"
#                      sources_text.append(f"[{source_id}] {doc}")
#                      source_mapping[source_id] = f"Document {i+1}" # Fallback name

#             fmt_context = "\n\n".join(sources_text)

#             prompt = f"""Analyze the following excerpts to answer the user's question.
#             Base your answer STRICTLY on the provided excerpts. Cite the source number (e.g., [Source 1], [Source 2]) where appropriate within the explanation and key points.

#             User Question: {query}

#             Excerpts:
#             {fmt_context}

#             ---
#             Provide your response ONLY as a JSON object with the following exact keys:
#             - "direct_answer": A concise, direct answer to the question.
#             - "detailed_explanation": A thorough explanation based on the excerpts, citing sources like [Source X].
#             - "key_points": A list of strings, where each string is a key takeaway point, citing sources like [Source X].

#             Example JSON format:
#             {{
#               "direct_answer": "The main finding is X.",
#               "detailed_explanation": "Excerpt [Source 1] states that... Additionally, [Source 2] mentions...",
#               "key_points": [
#                 "Fact A is supported by [Source 1].",
#                 "Fact B is mentioned in [Source 2].",
#                 "Conclusion C can be drawn from [Source 1] and [Source 2]."
#               ]
#             }}

#             Generate the JSON response now:
#             """

#             response = self.llm.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are an assistant that analyzes provided text excerpts and answers questions based ONLY on them. You output answers in a specific JSON format."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=temperature,
#                 response_format={"type": "json_object"} # Request JSON output
#             )

#             raw_response_content = response.choices[0].message.content.strip()

#             # --- Parse the JSON response ---
#             try:
#                 structured_answer = json.loads(raw_response_content)
#                 # Basic validation
#                 if not all(k in structured_answer for k in ["direct_answer", "detailed_explanation", "key_points"]):
#                     print("Warning: LLM JSON response missing expected keys. Falling back to raw text.")
#                     return f"LLM Response (unexpected format):\n```json\n{raw_response_content}\n```" # Return raw if keys missing

#                 # Store source mapping for UI display later
#                 structured_answer["source_mapping"] = source_mapping
            
            
#                 if check_for_hallucinations:
#                     # Check main parts - combine them for a single check or check individually?
#                     # Let's check the detailed explanation as it's likely the most comprehensive part.
#                     hall_check_text = structured_answer.get("detailed_explanation", "")
#                     if hall_check_text:
#                         hall_res = self._check_hallucination(query, raw_docs, hall_check_text)
#                         if hall_res and not hall_res.get("is_grounded", True):
#                              warnings = "\n".join([f"- \"{s.get('statement','N/A')}\" ({s.get('reason','N/A')})" for s in hall_res.get("unsupported_statements", [])])
#                              # Add warning to the dictionary - UI needs to know how to display it
#                              structured_answer["hallucination_warning"] = f"**⚠️ Warning:** The detailed explanation might contain unsupported statements:\n{warnings}"

#                 return structured_answer # Return the parsed dictionary

#             except json.JSONDecodeError as json_e:
#                 print(f"Error decoding LLM JSON response: {json_e}")
#                 print(f"Raw response was: {raw_response_content}")
#                 # Fallback: return the raw response string if JSON parsing fails
#                 return f"Couldn't parse the structured answer. Raw response:\n```\n{raw_response_content}\n```"
#             except Exception as parse_e:
#                 print(f"Error processing LLM response: {parse_e}")
#                 return f"Error processing response: {parse_e}"


#         except Exception as e:
#             st.error(f"Generate response error: {e}")
#             print(f"Generate response error: {e}")
#             return f"An error occurred during response generation: {e}" # Return error string


#     def get_embedding_info(self) -> Dict[str, Any]:
#         model_selector=TheModelSelector(); model_info=model_selector.embedding_models[self.embedding_model_name]
#         return {"name": model_info["name"], "dimensions": model_info["dimensions"], "model": self.embedding_model_name}

#     def get_collection_stats(self) -> Dict[str, Any]:
#         try:
#             if not self.collection: return {"count": 0, "sources": []}
#             count = self.collection.count(); sources = set()
#             limit = 10000; items = self.collection.get(limit=limit, include=['metadatas'])
#             if items and items.get("metadatas"): sources = {m["source"] for m in items["metadatas"] if m and "source" in m}
#             if count > limit and len(sources) < count : st.sidebar.caption(f"Source list from first {limit} chunks.") # Refined warning
#             return {"count": count, "sources": sorted(list(sources))}
#         except Exception as e: st.error(f"Get stats error: {e}"); print(f"Get stats error: {e}"); return {"count": 0, "sources": []}

#     def delete_document_by_source(self, source_name: str) -> bool:
#         """Deletes chunks by source and updates index."""
#         # ... [Same implementation as before] ...
#         if not source_name or not self.collection: return False
#         try:
#             st.info(f"Finding chunks for '{source_name}'...")
#             res = self.collection.get(where={"source": source_name}, include=[])
#             ids_to_del = res.get("ids")
#             if not ids_to_del: st.warning(f"No chunks found for '{source_name}'."); return False
#             st.info(f"Deleting {len(ids_to_del)} chunks for '{source_name}'...")
#             self.collection.delete(ids=ids_to_del)
#             print(f"Deleted '{source_name}'. Updating index..."); time.sleep(0.5); self._update_keyword_search_index_from_db()
#             return True # Let UI handle success message
#         except Exception as e: st.error(f"Delete error for '{source_name}': {e}"); print(f"Delete error: {e}"); return False


#     def reset_collection(self) -> bool:
#         """Resets collection and index."""
#         # ... [Same implementation as before] ...
#         try:
#             if not self.collection: return False
#             count = self.collection.count()
#             if count == 0: st.info("Collection empty."); return True
#             st.warning(f"Resetting '{self.collection_name}' ({count} items)...")
#             self.db.delete_collection(name=self.collection_name)
#             self.collection = self._setup_collection()
#             print("Collection reset. Updating index..."); self._update_keyword_search_index_from_db()
#             return True # Let UI handle success message
#         except Exception as e: st.error(f"Reset error: {e}"); print(f"Reset error: {e}"); return False



# # --- Streamlit UI (3 Tabs so far: Chat, Upload, View) ---
# def main():
#     st.set_page_config(page_title="SmartBot Doc Q&A", page_icon="📚", layout="wide")
#     st.title("📚 SmartBot Doc Q&A")
#     st.caption("Upload Documents, ask questions, and get answers grounded in your documents.")

#     # Session state initialization
#     if "rag_system" not in st.session_state: st.session_state.rag_system = None
#     if "upload_key_counter" not in st.session_state: st.session_state.upload_key_counter = 0
#     if "chat_history" not in st.session_state: st.session_state.chat_history = []
#     if "confirming_delete" not in st.session_state: st.session_state.confirming_delete = None # For delete confirmation
#     if "upload_success_message" not in st.session_state: st.session_state.upload_success_message = None # For upload feedback

#     # Setup RAG system (with spinner for initial load)
#     try:
#         if st.session_state.rag_system is None:
#             with st.spinner("Initializing Knowledge Base..."):
#                 model_selector = TheModelSelector()
#                 llm_model, embedding_model = model_selector.get_models()
#                 st.session_state.rag_system = TheRAGSystem(embedding_model, llm_model)
#             # No rerun here needed if sidebar updates dynamically or on next interaction
#         rag_system = st.session_state.rag_system # Assign for easier access

#         # Sidebar Info
#         with st.sidebar:
#             st.header("📊 Collection Status")
#             stats = rag_system.get_collection_stats()
#             st.metric("Total Chunks", stats.get("count", 0))
#             st.metric("Documents", len(stats.get("sources", [])))
#             st.divider()
#             st.header("⚙️ Config")
#             model_selector = TheModelSelector() # Re-instantiate for latest info if needed
#             embedding_info = model_selector.embedding_models["openai"]
#             st.info(f"LLM: GPT-4o-mini\nEmbeddings: {embedding_info['name']}\nDimensions: {embedding_info['dimensions']}")
#             st.info("Retrieval: Hybrid Search\n(Semantic + Keyword)")

#     except Exception as e:
#         st.error(f"Fatal error initializing RAG system: {e}")
#         print(f"Fatal error initializing RAG system: {e}")
#         st.stop() # Stop execution if RAG fails

#     # --- Define Tabs ---
#     tab_chat, tab_upload, tab_view = st.tabs(["💬 Chat", "➕ Upload Documents", "📄 Manage Documents"])

#     # --- Chat Tab ---
#     with tab_chat:
#         st.header("Ask Questions")
#         stats = rag_system.get_collection_stats() # Get current stats

#         if stats["count"] == 0:
#             st.info("Please upload documents in the 'Upload Documents' tab first.")
#         else:
#             col_context, col_temp = st.columns([3, 1])
#             with col_context:
#                 sources_list = ["All Documents"] + stats.get("sources", [])
#                 selected_context = st.selectbox("Chat with:", options=sources_list, key="chat_context_select")
#             with col_temp:
#                 temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1, key="chat_temp_slider", help="Think of temperature like a creativity dial. At 0.0 (cold), the AI gives consistent, focused answers - great for factual queries. At 1.0 (hot), it gets more creative and varied - better for brainstorming. Example: For 'What is quantum computing?', 0.2 gives technical definitions, while 0.8 might include analogies and examples.")

#             query_filter = {"source": selected_context} if selected_context != "All Documents" else None
#             if query_filter: st.caption(f"Querying within: **{selected_context}**")

#             # Simple Chat History Display
#             st.markdown("---")
#             st.subheader("Chat History")
#             if not st.session_state.chat_history: st.caption("No questions asked yet.")
#             # Display history - assumes content is already formatted string
#             for message in st.session_state.chat_history:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"]) # Display pre-formatted content

#             # Chat Input & Processing
#             user_query = st.chat_input("Ask something about the documents...")
#             if user_query:
#                 st.session_state.chat_history.append({"role": "user", "content": user_query})
#                 with st.chat_message("user"): st.markdown(user_query)

#                 with st.chat_message("assistant"):
#                     placeholder = st.empty(); placeholder.markdown("Thinking... 🤔")
#                     formatted_response_content = "" # String to hold the final formatted display text
#                     try:
#                         start_time = time.time()
#                         context = rag_system.query_documents(query=user_query, n_results=3, where_filter=query_filter)
#                         response = rag_system.generate_response(query=user_query, context=context, temperature=temperature, check_for_hallucinations=True)
#                         end_time = time.time()
#                         print(f"Chat query took {end_time - start_time:.2f}s")

#                         # --- START: Formatting Logic ---
#                         if isinstance(response, dict):
#                             # It's the structured dictionary, format it nicely
#                             display_parts = []

#                             # 1. Add Hallucination Warning (if present)
#                             if "hallucination_warning" in response:
#                                 display_parts.append(response["hallucination_warning"]) # Should already be markdown formatted

#                             # 2. Add Direct Answer
#                             if response.get("direct_answer"):
#                                 display_parts.append(f"**Answer:** {response['direct_answer']}")

#                             # 3. Add Detailed Explanation
#                             if response.get("detailed_explanation"):
#                                 # Add a newline before if other parts exist
#                                 prefix = "\n" if display_parts else ""
#                                 display_parts.append(f"{prefix}**Explanation:**\n{response['detailed_explanation']}")

#                             # 4. Add Key Points
#                             if response.get("key_points"):
#                                 points_markdown = "\n".join([f"- {point}" for point in response["key_points"]])
#                                 # Add a newline before if other parts exist
#                                 prefix = "\n" if display_parts else ""
#                                 display_parts.append(f"{prefix}**Key Points:**\n{points_markdown}")

#                             # Combine all parts into a single string
#                             formatted_response_content = "\n\n".join(display_parts)

#                         elif isinstance(response, str):
#                             # It's already a string (e.g., error message, fallback)
#                             formatted_response_content = response
#                         else:
#                             # Handle unexpected response types
#                             formatted_response_content = "Sorry, I received an unexpected response format."
#                             print(f"Unexpected response type: {type(response)}")

#                         placeholder.markdown(formatted_response_content) # Display the formatted string
#                         # --- END: Formatting Logic ---

#                         # Optional: Keep the source expander - it shows the raw context provided *to* the LLM
#                         sources = context.get("documents", [[]])[0] if context else []
#                         metadatas = context.get("metadatas", [[]])[0] if context else []
#                         if sources and metadatas:
#                             with st.expander("View Raw Sources Used in Answer Generation", expanded=False):
#                                 source_mapping = response.get("source_mapping", {}) if isinstance(response, dict) else {}
#                                 # Create a reverse map for display if needed, or just use index
#                                 source_name_map = {v: k for k, v in source_mapping.items()} # Map "file.pdf" -> "Source 1"

#                                 for idx, (doc, meta) in enumerate(zip(sources, metadatas)):
#                                     source_file = meta.get("source", "Unknown")
#                                     # Try to find the citation number used in the text
#                                     citation_label = source_name_map.get(source_file, f"Source {idx+1}")
#                                     st.markdown(f"**[{citation_label}] `{source_file}`**")
#                                     display_doc = doc[:500] + "..." if len(doc) > 500 else doc
#                                     st.text_area(f"snippet_chat_{idx}", display_doc, height=100, disabled=True, label_visibility="collapsed")
#                         elif not context or not context.get("documents") or not context["documents"][0]:
#                             st.caption("Note: No relevant document excerpts were found to base the answer on.")


#                     except Exception as chat_e:
#                         formatted_response_content = f"An error occurred during response generation: {chat_e}"
#                         placeholder.error(formatted_response_content)
#                         print(f"Chat processing error: {chat_e}")

#                 # Store the *formatted* assistant response string in history
#                 st.session_state.chat_history.append({"role": "assistant", "content": formatted_response_content})


#             # Button to clear chat history
#             st.markdown("---")
#             if st.button("Clear Chat History"):
#                 st.session_state.chat_history = []
#                 st.rerun()

#     # --- Upload Tab ---
#     with tab_upload:
#         # ... (rest of the Upload Tab code remains the same) ...
#         st.header("Upload New Document")
#         st.markdown("Select a PDF, DOCX, or TXT file to process and add to the knowledge base.")

#         # Advanced Options Expander
#         with st.expander("Advanced Processing Options"):
#             chunk_size_opt = st.slider("Target Chunk Size (chars)", 300, 2000, CHUNK_SIZE, 100, key="upload_chunk_size", help="Think of this like breaking a big book into smaller sections. A larger chunk size means bigger sections (more context but might be less precise), while smaller chunks mean shorter sections (more precise but might miss context). Example: with 1000, a page might be split into 3-4 parts.")
#             chunk_overlap_opt = st.slider("Chunk Overlap (chars)", 0, 500, CHUNK_OVERLAP, 50, key="upload_chunk_overlap", help="This is like having each section share a few sentences with the next section, so we don't lose the connection between ideas. Example: if overlap is 200, each chunk will share about 200 characters with the next chunk to maintain context.")

#         uploaded_file = st.file_uploader(
#             "Choose a file",
#             type=["pdf", "docx", "txt"], # Accept multiple types
#             accept_multiple_files=False, # Process one at a time
#             key=f"file_uploader_{st.session_state.upload_key_counter}" # Dynamic key
#             )

#         if uploaded_file is not None:
#             st.markdown("---")
#             st.write(f"**File:** {uploaded_file.name}")
#             st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

#             # Check if already exists
#             existing_sources = rag_system.get_collection_stats().get("sources", set())
#             process_button_label = "Process & Add Document"
#             if uploaded_file.name in existing_sources:
#                 st.warning(f"⚠️ '{uploaded_file.name}' seems to be already indexed.")
#                 process_button_label = "Process Anyway"

#             if st.button(process_button_label, type="primary", key=f"process_btn_{st.session_state.upload_key_counter}"):
#                 with st.spinner(f"Processing '{uploaded_file.name}'... Please wait."):
#                     try:
#                         processor = TheDocProcessor(chunk_size=chunk_size_opt, chunk_overlap=chunk_overlap_opt)
#                         read_start = time.time()
#                         document_text = processor.read_document(uploaded_file, uploaded_file.name)
#                         read_time = time.time() - read_start
#                         if not document_text: st.error("Failed to extract text."); st.stop() # Stop if no text

#                         st.info(f"Read document ({len(document_text)} chars) in {read_time:.2f}s.")
#                         chunking_start = time.time()
#                         chunks = processor.create_chunks(document_text, uploaded_file)
#                         chunking_time = time.time() - chunking_start
#                         if not chunks: st.error("Failed to create chunks."); st.stop() # Stop if no chunks

#                         st.info(f"Created {len(chunks)} chunks in {chunking_time:.2f}s. Adding to DB...")
#                         add_start = time.time()
#                         success = rag_system.add_documents(chunks)
#                         add_time = time.time() - add_start

#                         if success:
#                             total_time = time.time() - read_start
#                             st.session_state.upload_success_message = f"✅ Successfully added '{uploaded_file.name}' ({len(chunks)} chunks) in {total_time:.2f}s."
#                             st.session_state.upload_key_counter += 1 # Increment key BEFORE rerun
#                             st.rerun() # Rerun to clear uploader and show message
#                         else:
#                             st.error("❌ Failed to add document chunks to the database.")
#                     except Exception as upload_e:
#                         st.error(f"An unexpected error occurred during processing: {upload_e}")
#                         print(f"Upload processing error: {upload_e}")

#         # Display success message from session state after rerun
#         if st.session_state.upload_success_message:
#             st.success(st.session_state.upload_success_message)
#             st.session_state.upload_success_message = None # Clear after displaying


#     # --- View Documents Tab ---
#     with tab_view:
#         st.header("View and Manage Uploaded Documents")
#         view_stats = rag_system.get_collection_stats()
#         view_sources = view_stats.get("sources", [])

#         if not view_sources:
#             st.info("No documents have been uploaded yet. Use the 'Upload Documents' tab to upload some documents first.")
#         else:
#             st.markdown(f"You have **{len(view_sources)}** document(s) indexed, comprising **{view_stats.get('count', 0)}** text chunks.")
#             st.markdown("---")

#             col1, col2 = st.columns([2,1]) # Layout for list and delete

#             with col1:
#                  st.subheader("📚 Indexed Documents")
#                  for i, src in enumerate(view_sources, 1): st.markdown(f"{i}. `{src}`")

#             with col2:
#                  st.subheader("🗑️ Delete Document")
#                  doc_to_delete = st.selectbox(
#                      "Select document to remove:", options=[""] + view_sources, index=0,
#                      key="view_doc_delete_select", help="Select a document to enable deletion.")

#                  delete_disabled = not bool(doc_to_delete)
#                  delete_btn_key = f"view_delete_doc_btn_{doc_to_delete}" if doc_to_delete else "view_delete_doc_btn_disabled"

#                  if st.button("Delete Selected Document", type="secondary", disabled=delete_disabled, key=delete_btn_key):
#                       if doc_to_delete:
#                            st.session_state.confirming_delete = doc_to_delete # Mark for confirmation
#                            st.rerun() # Rerun to show confirmation dialog

#                  elif delete_disabled: st.caption("Select a document to enable deletion.")

#                  # --- Confirmation Dialog Logic ---
#                  if st.session_state.confirming_delete:
#                      doc_name = st.session_state.confirming_delete
#                      st.warning(f"**Confirm Deletion:** Permanently remove '{doc_name}' and all its associated data?")
#                      confirm_col, cancel_col = st.columns(2)
#                      with confirm_col:
#                          if st.button(f"Confirm Delete '{doc_name}'", type="primary", key=f"confirm_del_{doc_name}"):
#                              with st.spinner(f"Deleting '{doc_name}'..."):
#                                  delete_success = rag_system.delete_document_by_source(doc_name)
#                                  st.session_state.confirming_delete = None # Clear confirmation state
#                                  if delete_success:
#                                      st.success(f"Successfully deleted '{doc_name}'.")
#                                      time.sleep(1.5) # Pause to show message
#                                  else:
#                                      st.error(f"Failed to delete '{doc_name}'. Check logs.")
#                                      time.sleep(2) # Pause to show message
#                                  st.rerun() # Rerun to update list/clear dialog
#                      with cancel_col:
#                          if st.button("Cancel", key=f"cancel_del_{doc_name}"):
#                               st.session_state.confirming_delete = None # Clear confirmation state
#                               st.rerun() # Rerun to hide dialog

#             # Optional: Add Reset Button
#             # st.markdown("---")
#             # if st.button("⚠️ Reset Entire Knowledge Base", type="secondary"):
#             #      if st.checkbox("Confirm you want to delete ALL indexed data?", key="reset_confirm_cb"):
#             #           if st.button("Confirm Reset", type="primary", key="reset_confirm_btn"):
#             #               with st.spinner("Resetting knowledge base..."):
#             #                   reset_success = rag_system.reset_collection()
#             #                   if reset_success: st.success("Knowledge base reset successfully.")
#             #                   else: st.error("Failed to reset knowledge base.")
#             #                   time.sleep(1.5)
#             #                   st.rerun()



# if __name__ == "__main__":
#     main()


import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import docx
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

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 4  # For parallel processing
MAX_SUMMARY_INPUT_CHARS = 20000 # Limit text sent for summary generation

# --- Model Selection ---
class TheModelSelector:
    """Handles AI model selection. Currently focused on OpenAI."""
    def __init__(self):
        self.llm_model = "openai" # Could be extended later
        self.embedding_model = "openai" # Could be extended later
        self.embedding_models = {
            "openai": {
                "name": "OpenAI Embeddings (text-embedding-3-small)",
                "dimensions": 1536,
                "model_name": "text-embedding-3-small",
            }
            # Add other embedding providers here if needed
        }
        self.llm_models = {
            "openai": {
                "qa_model": "gpt-4o-mini",
                "summary_model": "gpt-4o-mini", # Use a fast model for summaries
                "hallucination_check_model": "gpt-4o-mini"
            }
            # Add other LLM providers here if needed
        }

    def get_models(self) -> Tuple[str, str]:
        """Returns the selected LLM and embedding model provider names."""
        return self.llm_model, self.embedding_model

    def get_embedding_info(self, model_key: str = "openai") -> Optional[Dict[str, Any]]:
         """Gets details for a specific embedding model provider."""
         return self.embedding_models.get(model_key)

    def get_llm_info(self, model_key: str = "openai") -> Optional[Dict[str, Any]]:
        """Gets details for a specific LLM provider."""
        return self.llm_models.get(model_key)

# --- Document Processing (Handles PDF, DOCX, TXT) ---
class TheDocProcessor:
    """Handles reading documents (PDF, DOCX, TXT) and splitting them into manageable chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
             st.warning(f"Chunk overlap ({self.chunk_overlap}) >= chunk size ({self.chunk_size}). Setting overlap to {self.chunk_size // 5}.")
             self.chunk_overlap = self.chunk_size // 5

    def read_document(self, uploaded_file: io.BytesIO, file_name: str) -> str:
        """
        Reads text content from PDF, DOCX, or TXT files.
        Returns the full text content as a string.
        """
        text = ""
        try:
            uploaded_file.seek(0) # Reset file pointer
            file_extension = os.path.splitext(file_name)[1].lower()
            st.info(f"Reading {file_extension.upper()} file: '{file_name}'...")

            if file_extension == ".pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                total_pages = len(reader.pages)
                progress_bar = None
                # Only show progress bar for reasonably large PDFs to avoid flicker
                if total_pages > 10:
                    progress_text = f"Reading PDF '{file_name}' ({total_pages} pages)..."
                    try: progress_bar = st.progress(0, text=progress_text)
                    except Exception: st.info(progress_text); progress_bar = None # Handle potential streamlit errors

                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text: text += page_text + "\n" # Add newline between pages
                    except Exception as e:
                        st.warning(f"Could not extract text from PDF page {i+1} of '{file_name}': {e}")
                    if progress_bar:
                        try: progress_bar.progress((i + 1) / total_pages, text=progress_text)
                        except Exception: progress_bar = None # Gracefully handle if progress bar fails

                if progress_bar:
                    try: progress_bar.empty()
                    except Exception: pass # Ignore errors emptying the bar

            elif file_extension == ".docx":
                document = docx.Document(uploaded_file)
                full_text = [para.text for para in document.paragraphs]
                text = "\n".join(full_text)

            elif file_extension == ".txt":
                content_bytes = uploaded_file.read()
                try:
                    text = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    st.warning("UTF-8 decoding failed for TXT file, trying latin-1...")
                    try:
                        text = content_bytes.decode('latin-1')
                    except Exception as decode_err:
                        st.error(f"Failed to decode TXT file '{file_name}' with UTF-8 or latin-1: {decode_err}")
                        return "" # Return empty string on failure

            else:
                st.error(f"Unsupported file type: '{file_extension}'. Please upload PDF, DOCX, or TXT.")
                return ""

            # Basic text cleaning (replace multiple whitespace chars with a single space)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                st.warning(f"No text content extracted from '{file_name}'. The file might be empty, image-based, or corrupted.")
            return text

        except PyPDF2.errors.PdfReadError:
            st.error(f"Could not read PDF '{file_name}'. The file might be corrupted or password-protected.")
            return ""
        except Exception as e:
            st.error(f"An error occurred while reading '{file_name}': {e}")
            print(f"Error reading {file_name}:")
            traceback.print_exc() # Print full traceback to console
            return ""

    def _process_doc_chunk_internal(self, chunk_data: Tuple[int, int, str, str]) -> Dict[str, Any]:
        """Helper function to process one piece of the document text, designed for parallel execution."""
        start, end, text, doc_name = chunk_data
        # Calculate effective start considering overlap (don't go before 0)
        effective_start = max(0, start - self.chunk_overlap) if start > 0 else 0

        # Initial chunk text based on effective start and desired end
        chunk_text = text[effective_start:end]

        final_end = end # Initialize final end position

        # Try to end chunk on sentence boundary if not the last chunk
        if end < len(text):
            # Find sentence endings (.!?) followed by whitespace near the desired end
            # Look within a reasonable window around the target end position
            search_start = max(0, len(chunk_text) - self.chunk_overlap - 50) # Look back a bit
            possible_ends = [m.start() + 1 for m in re.finditer(r'[.!?]\s', chunk_text[search_start:])]

            if possible_ends:
                # Find the ending closest to the original 'end' position within the chunk
                ideal_end_in_chunk = end - effective_start
                best_end_in_chunk = -1
                min_diff = float('inf')

                for p_end_relative in possible_ends:
                    p_end_absolute_in_chunk = search_start + p_end_relative
                    # Only consider ends that are reasonably close to the target chunk size
                    # And ensure we don't make the chunk excessively long or short accidentally
                    if abs(p_end_absolute_in_chunk - ideal_end_in_chunk) < self.chunk_size * 0.4: # Allow deviation up to 40%
                         diff = abs(p_end_absolute_in_chunk - ideal_end_in_chunk)
                         if diff < min_diff:
                            min_diff = diff
                            best_end_in_chunk = p_end_absolute_in_chunk

                if best_end_in_chunk != -1:
                    final_end = effective_start + best_end_in_chunk
                    chunk_text = text[effective_start:final_end] # Recalculate chunk text
        else:
             # This is the last chunk, ensure it goes to the very end
             final_end = len(text)
             chunk_text = text[effective_start:final_end]

        # Clean up whitespace and create the result dictionary
        chunk_text = chunk_text.strip()
        return {
            "id": str(uuid.uuid4()),
            "text": chunk_text,
            "metadata": {"source": doc_name, "start": effective_start, "end": final_end},
            "end_pos": final_end # Pass the actual end position for the next iteration
        }


    def create_chunks(self, text: str, file_obj: io.BytesIO) -> List[Dict[str, Any]]:
        """Splits document text into chunks with overlap and sentence boundary awareness."""
        if not text: return []
        chunks = []; start = 0; doc_len = len(text); file_name = file_obj.name

        # Determine processing method based on document size (heuristic)
        # Larger documents benefit more from parallel processing overhead
        use_parallel = doc_len > 150000 and MAX_WORKERS > 1 # Only use parallel if doc is large enough and workers > 1

        if not use_parallel:
            # --- Sequential Chunking ---
            print(f"Using sequential chunking for '{file_name}'...")
            current_pos = 0
            while current_pos < doc_len:
                end = min(current_pos + self.chunk_size, doc_len)
                chunk_data = (current_pos, end, text, file_name)
                processed_chunk = self._process_doc_chunk_internal(chunk_data)

                # Add chunk if it has content
                if processed_chunk["text"]:
                    chunks.append({
                        "id": processed_chunk["id"],
                        "text": processed_chunk["text"],
                        "metadata": processed_chunk["metadata"]
                    })

                # Move to the next position based on the actual end of the processed chunk
                next_start = processed_chunk["end_pos"]

                # Safety break: If we didn't advance, force advancement by a small step
                # This prevents infinite loops if overlap logic somehow stalls
                if next_start <= current_pos:
                    print(f"Warning: Chunking stalled at position {current_pos}. Advancing.")
                    next_start = current_pos + 1

                current_pos = next_start

        else:
            # --- Parallel Chunking ---
            st.info(f"Using parallel chunking for '{file_name}' (Length: {doc_len} chars)...")
            tasks = []
            current_pos = 0
            while current_pos < doc_len:
                 end = min(current_pos + self.chunk_size, doc_len)
                 # Create tasks based on initial start/end, overlap handled within the function
                 tasks.append((current_pos, end, text, file_name))
                 # Prepare start for the *next* theoretical chunk before overlap adjustment
                 current_pos = end

            total_tasks = len(tasks); prog_bar = None; prog_txt=f"Chunking '{file_name}'... (0/{total_tasks} tasks)"
            try:
                prog_bar = st.progress(0, text=prog_txt)
            except Exception:
                st.info(prog_txt); prog_bar = None # Fallback if progress bar fails

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(self._process_doc_chunk_internal, task): task for task in tasks}
                count_completed = 0
                # Process as tasks complete
                for future in concurrent.futures.as_completed(future_to_task):
                    task_data = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        count_completed += 1
                        if prog_bar:
                           try:
                               prog_bar.progress(count_completed / total_tasks, text=f"Chunking '{file_name}'... ({count_completed}/{total_tasks} tasks)")
                           except Exception: prog_bar = None # Handle progress bar failure
                    except Exception as exc:
                        st.warning(f"Chunk generation failed for part starting near {task_data[0]}: {exc}")
                        print(f"Chunk failed for task {task_data}: {exc}")

            if prog_bar:
                 try: prog_bar.empty()
                 except Exception: pass

            # Sort results by their starting position before assembling final chunks
            results.sort(key=lambda r: r['metadata']['start'])

            # Filter out empty chunks and format
            chunks = [{"id": r["id"], "text": r["text"], "metadata": r["metadata"]} for r in results if r["text"]]

        st.write(f"Generated {len(chunks)} chunks for '{file_name}'.")
        return chunks


class TheRAGSystem:
    """
    Stores documents, performs hybrid search (semantic + keyword),
    generates answers, and manages document summaries.
    """

    def __init__(self, embedding_model_provider="openai", llm_provider="openai"):
        self.embedding_model_provider = embedding_model_provider
        self.llm_provider = llm_provider
        self.db_path = "./chroma_db"
        self.collection_name = f"documents_{self.embedding_model_provider}" # Collection name depends on embedding

        # Setup Models, DB, Embedding Fn, LLM (with error handling)
        self.model_selector = TheModelSelector()
        self.llm_info = self.model_selector.get_llm_info(self.llm_provider)
        self.emb_info = self.model_selector.get_embedding_info(self.embedding_model_provider)

        if not self.llm_info or not self.emb_info:
            st.error("Failed to load model configurations.")
            st.stop()

        # Database Client
        try:
            self.db = chromadb.PersistentClient(path=self.db_path)
        except Exception as e:
            st.error(f"Fatal Error: Could not initialize ChromaDB client at '{self.db_path}': {e}")
            st.stop()

        # OpenAI API Key Check
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("Fatal Error: OPENAI_API_KEY environment variable not found.")
            st.stop()

        # Embedding Function
        try:
            # Currently only supports OpenAI embeddings
            if self.embedding_model_provider == "openai":
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=self.openai_api_key,
                    model_name=self.emb_info['model_name']
                )
            else:
                raise NotImplementedError(f"Embedding provider '{self.embedding_model_provider}' not supported yet.")
        except Exception as e:
            st.error(f"Fatal Error: Could not initialize embedding function: {e}")
            st.stop()

        # LLM Client
        try:
            # Currently only supports OpenAI LLM
            if self.llm_provider == "openai":
                 self.llm = OpenAI(api_key=self.openai_api_key)
            else:
                 raise NotImplementedError(f"LLM provider '{self.llm_provider}' not supported yet.")
        except Exception as e:
            st.error(f"Fatal Error: Could not initialize OpenAI LLM client: {e}")
            self.llm = None # Set LLM to None if initialization fails
            # Depending on criticality, you might st.stop() here too

        # Chroma Collection Setup
        self.collection = self._setup_collection()

        # BM25 Index Components (for keyword search)
        self.corpus: List[str] = []         # List of document chunk texts
        self.doc_ids: List[str] = []        # List of corresponding chunk IDs
        self.doc_metadatas: List[Dict] = [] # List of corresponding chunk metadatas
        self.bm25: Optional[BM25Okapi] = None # The BM25 index object
        self._update_keyword_search_index_from_db() # Initialize BM25 index

    def _setup_collection(self) -> Optional[chromadb.Collection]:
        """Gets or creates the ChromaDB collection."""
        try:
            # Using cosine distance for embeddings, common practice
            collection = self.db.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn, # type: ignore [arg-type] # Ignore type hint issue for now
                metadata={"hnsw:space": "cosine"} # Use cosine distance
            )
            print(f"Successfully accessed or created collection: '{self.collection_name}'")
            return collection
        except Exception as e:
            st.error(f"Fatal Error: Could not get or create ChromaDB collection '{self.collection_name}': {e}")
            print(f"Collection setup error for '{self.collection_name}':")
            traceback.print_exc()
            st.stop() # Stop execution if collection cannot be setup
            return None # Should not be reached due to st.stop()

    def _update_keyword_search_index_from_db(self):
        """Fetches all documents from Chroma and rebuilds the BM25 index."""
        print("Attempting to update keyword (BM25) index...")
        if not self.collection:
            print("Warning: BM25 update skipped, collection not available.")
            return

        try:
            # Fetch all data needed for BM25: IDs, documents (text), metadatas
            # Fetch in batches if the collection is huge? For now, fetch all.
            # Consider adding limits/paging if collection grows extremely large.
            all_data = self.collection.get(include=["metadatas", "documents"]) # IDs are included by default

            if not all_data or not all_data.get("ids"):
                print("Keyword index reset: No documents found in the collection.")
                self.corpus = []
                self.doc_ids = []
                self.doc_metadatas = []
                self.bm25 = None
                return

            self.doc_ids = all_data["ids"]
            self.corpus = all_data["documents"] or [] # Ensure corpus is a list even if documents are None/empty
            self.doc_metadatas = all_data["metadatas"] or [] # Ensure metadatas is a list

            if not self.corpus:
                print("Warning: No document text found for BM25 index, clearing index.")
                self.bm25 = None
                return

            # Basic tokenization for BM25 (lowercase, split by space)
            # More advanced tokenization (e.g., removing punctuation, stemming) could be added here.
            print(f"Building BM25 index for {len(self.corpus)} document chunks...")
            start_time = time.time()
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            end_time = time.time()
            print(f"BM25 index built successfully in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            st.error(f"Error updating keyword search index: {e}")
            print(f"Keyword (BM25) index update failed:")
            traceback.print_exc()
            self.bm25 = None # Invalidate index on error

    def _keyword_search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> Dict[str, List]:
        """Performs BM25 keyword search on the indexed corpus."""
        if not self.bm25 or not self.corpus or not self.doc_ids:
            print("Keyword search skipped: BM25 index or corpus is not available.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]} # Match Chroma format

        try:
            # Tokenize query similarly to how the corpus was tokenized
            tokens = query.lower().split()
            if not tokens:
                 return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # Get BM25 scores for the query against all documents
            scores = self.bm25.get_scores(tokens)

            # Filter results based on the 'where_filter' (applied *after* scoring)
            results_with_scores = []
            for i, score in enumerate(scores):
                # Basic safety check for list lengths
                if i >= len(self.doc_metadatas) or i >= len(self.doc_ids) or i >= len(self.corpus):
                    print(f"Warning: Index mismatch during keyword search at index {i}.")
                    continue

                metadata = self.doc_metadatas[i]
                doc_id = self.doc_ids[i]
                document_text = self.corpus[i]

                # Apply the where filter if provided
                if where_filter:
                    match = True
                    for key, value in where_filter.items():
                        if str(metadata.get(key)) != str(value): # Simple string comparison
                            match = False
                            break
                    if not match:
                        continue # Skip this document if filter doesn't match

                # Store index, id, score, metadata, and document text
                if score > 0: # Only consider documents with a positive BM25 score
                    results_with_scores.append((i, doc_id, score, metadata, document_text))

            # Sort results by score in descending order
            results_with_scores.sort(key=lambda x: x[2], reverse=True)

            # Select top N results
            top_results = results_with_scores[:n_results]

            # Format results similar to Chroma's output
            final_ids = [r[1] for r in top_results]
            final_docs = [r[4] for r in top_results] # Include document text
            final_metas = [r[3] for r in top_results]

            # Normalize BM25 scores to pseudo-distances (0=best, 1=worst) for consistency
            # This is a simple normalization; more sophisticated methods exist.
            final_dists = []
            if top_results:
                scores_only = [r[2] for r in top_results]
                max_score = max(scores_only) if scores_only else 1.0
                min_score = min(scores_only) if scores_only else 0.0
                score_range = max_score - min_score if max_score > min_score else 1.0

                # Normalize score to distance: dist = 1 - (score - min) / range
                final_dists = [max(0.0, 1.0 - ((r[2] - min_score) / score_range)) if score_range > 0 else 0.5 for r in top_results]

            return {"ids": [final_ids], "documents": [final_docs], "metadatas": [final_metas], "distances": [final_dists]}

        except Exception as e:
            st.warning(f"Keyword search failed: {e}")
            print(f"Keyword search error:")
            traceback.print_exc()
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]} # Return empty on error

    def _hybrid_search(self, query: str, n_results: int = 5, where_filter: Optional[Dict] = None) -> Dict[str, List]:
        """Performs hybrid search using Reciprocal Rank Fusion (RRF) of semantic and keyword results."""
        print(f"Performing Hybrid Search (RRF) for query: '{query[:50]}...'")
        if not self.collection:
            st.error("Hybrid search failed: Collection is not available.")
            return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}

        try:
            # 1. Perform Semantic Search (ChromaDB vector search)
            # Fetch more results initially to improve fusion potential
            semantic_n = max(n_results * 2, 10)
            print(f"  - Semantic search (top {semantic_n})...")
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=semantic_n,
                where=where_filter,
                include=["metadatas", "documents", "distances"] # Ensure documents are included
            )

            # 2. Perform Keyword Search (BM25)
            keyword_n = max(n_results * 2, 10)
            print(f"  - Keyword search (top {keyword_n})...")
            keyword_results = self._keyword_search(query, n_results=keyword_n, where_filter=where_filter)

            # 3. Combine results using Reciprocal Rank Fusion (RRF)
            # RRF Score = sum(1 / (k + rank)) for each document across result sets
            # k is a constant, often 60, balances influence of lower-ranked items

            k = 60 # RRF constant
            fused_scores: Dict[str, float] = {}
            doc_details: Dict[str, Dict[str, Any]] = {} # Store doc text and metadata by ID

            # Process semantic results
            if semantic_results and semantic_results.get("ids") and semantic_results["ids"][0]:
                sem_ids = semantic_results["ids"][0]
                sem_docs = semantic_results["documents"][0]
                sem_metas = semantic_results["metadatas"][0]
                print(f"    - Processing {len(sem_ids)} semantic results.")
                for rank, doc_id in enumerate(sem_ids):
                    if doc_id:
                        score = 1 / (k + rank + 1) # Rank is 0-based
                        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score
                        if doc_id not in doc_details:
                            # Safely get details, assuming lists are aligned
                            doc_text = sem_docs[rank] if rank < len(sem_docs) else "N/A"
                            doc_meta = sem_metas[rank] if rank < len(sem_metas) else {}
                            doc_details[doc_id] = {"doc": doc_text, "meta": doc_meta}

            # Process keyword results
            if keyword_results and keyword_results.get("ids") and keyword_results["ids"][0]:
                key_ids = keyword_results["ids"][0]
                key_docs = keyword_results["documents"][0]
                key_metas = keyword_results["metadatas"][0]
                print(f"    - Processing {len(key_ids)} keyword results.")
                for rank, doc_id in enumerate(key_ids):
                     if doc_id:
                        score = 1 / (k + rank + 1)
                        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score
                        if doc_id not in doc_details:
                            # Safely get details
                            doc_text = key_docs[rank] if rank < len(key_docs) else "N/A"
                            doc_meta = key_metas[rank] if rank < len(key_metas) else {}
                            doc_details[doc_id] = {"doc": doc_text, "meta": doc_meta}

            if not fused_scores:
                print("  - No combined results after fusion.")
                return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}

            # Sort fused results by RRF score (higher is better)
            sorted_fused_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
            print(f"  - Combined {len(sorted_fused_results)} unique results via RRF.")

            # Select top N results
            top_n_fused = sorted_fused_results[:n_results]

            # Format final results like Chroma
            final_ids = [doc_id for doc_id, score in top_n_fused]
            final_docs = [doc_details.get(doc_id, {}).get("doc", "N/A") for doc_id in final_ids]
            final_metas = [doc_details.get(doc_id, {}).get("meta", {}) for doc_id in final_ids]
            # Hybrid search doesn't have a natural distance, return dummy or RRF scores?
            # Let's return normalized RRF scores as pseudo-distances (0=best) for consistency.
            final_dists = []
            if top_n_fused:
                scores_only = [score for _, score in top_n_fused]
                max_s = max(scores_only) if scores_only else 1.0
                min_s = min(scores_only) if scores_only else 0.0
                range_s = max_s - min_s if max_s > min_s else 1.0
                # Normalize score to distance: dist = 1 - (score - min) / range
                final_dists = [max(0.0, 1.0 - ((score - min_s) / range_s)) if range_s > 0 else 0.5 for _, score in top_n_fused]


            print(f"  - Hybrid search returning top {len(final_ids)} results.")
            return {"ids": [final_ids], "documents": [final_docs], "metadatas": [final_metas], "distances": [final_dists]}

        except Exception as e:
            st.error(f"Hybrid search encountered an error: {e}")
            print(f"Hybrid search error:")
            traceback.print_exc()
            st.warning("Falling back to semantic search only due to hybrid search error.")
            # Fallback to pure semantic search
            try:
                return self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_filter,
                    include=["metadatas", "documents", "distances"]
                )
            except Exception as fb_e:
                st.error(f"Fallback semantic search also failed: {fb_e}")
                print(f"Fallback semantic search error:")
                traceback.print_exc()
                return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]} # Final fallback empty

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Adds document chunks to the ChromaDB collection and updates the keyword index."""
        if not chunks:
            st.info("No chunks provided to add.")
            return True # Nothing to add, operation considered successful

        if not self.collection:
            st.error("Cannot add documents: Collection is not available.")
            return False

        try:
            ids = [c["id"] for c in chunks]
            docs = [c["text"] for c in chunks]
            metas = [c["metadata"] for c in chunks]

            # Batch adding for potentially large number of chunks
            batch_size = 100 # ChromaDB recommendation is often around 100-500
            total_chunks = len(chunks)
            num_batches = math.ceil(total_chunks / batch_size)

            prog_bar = None
            if total_chunks > batch_size: # Only show progress bar if multiple batches
                progress_text = f"Adding {total_chunks} chunks to knowledge base..."
                try: prog_bar = st.progress(0, text=progress_text)
                except Exception: st.info(progress_text); prog_bar = None

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_chunks)
                batch_ids = ids[start_idx:end_idx]
                batch_docs = docs[start_idx:end_idx]
                batch_metas = metas[start_idx:end_idx]

                if not batch_ids: continue # Skip empty batch

                # Add batch to Chroma
                self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

                if prog_bar:
                    progress = (i + 1) / num_batches
                    text = f"Adding batch {i+1}/{num_batches} ({end_idx}/{total_chunks} chunks)..."
                    try: prog_bar.progress(progress, text=text)
                    except Exception: prog_bar = None # Handle error

            if prog_bar:
                try: prog_bar.empty()
                except Exception: pass

            print(f"Successfully added {total_chunks} chunks. Updating keyword index...")
            # Update the keyword search index after adding new documents
            self._update_keyword_search_index_from_db()
            return True

        except Exception as e:
            st.error(f"Failed to add document chunks: {e}")
            print(f"Add documents error:")
            traceback.print_exc()
            return False

    def query_documents(self, query: str, n_results: int = 3, where_filter: Optional[Dict] = None) -> Optional[Dict[str, List]]:
        """Searches documents using HYBRID SEARCH (RRF)."""
        if not self.collection:
            st.error("Cannot query documents: Collection is not available.")
            return None
        if not query:
            st.warning("Query cannot be empty.")
            return None

        try:
            start_time = time.time()
            results = self._hybrid_search(query, n_results=n_results, where_filter=where_filter)
            end_time = time.time()
            print(f"Hybrid query execution time: {end_time - start_time:.2f}s")

            # Check if results are valid and contain documents
            if results and isinstance(results.get("documents"), list) and results["documents"] and results["documents"][0]:
                return results
            elif results:
                # Hybrid search returned a valid structure, but no documents matched
                print("Hybrid search completed, but found no matching documents for the query.")
                return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]} # Return empty but valid structure
            else:
                # Hybrid search itself failed or returned invalid structure (should be handled internally, but belt-and-suspenders)
                st.error("Hybrid search returned an unexpected result.")
                return None
        except Exception as e:
            st.error(f"An error occurred during document query: {e}")
            print(f"Query documents error:")
            traceback.print_exc()
            return None

    def _check_hallucination(self, query: str, context_docs: list, generated_answer: str) -> Optional[Dict]:
        """
        Uses an LLM to check if the generated answer is grounded in the provided context documents.
        Returns a dictionary with 'is_grounded' and potentially 'unsupported_statements', or None on error.
        """
        if not self.llm:
            st.warning("Hallucination check skipped: LLM client is not available.")
            return None

        try:
            # Prepare context string
            context_str = "\n\n---\n\n".join(f"Document Snippet {i+1}:\n{doc}" for i, doc in enumerate(context_docs))
            # Limit context length to avoid exceeding token limits for the check prompt
            max_context_len = 15000
            if len(context_str) > max_context_len:
                context_str = context_str[:max_context_len] + "\n... (context truncated)"

            # Define the prompt for the hallucination check LLM
            prompt = f"""Please act as a fact-checker. Your task is to determine if the 'Generated Answer' below is FULLY supported ONLY by the information present in the 'Context Documents'. Do not use any external knowledge.

            User's Query:
            {query}

            Context Documents:
            {context_str}
            ---
            Generated Answer:
            {generated_answer}
            ---
            Analyze the 'Generated Answer' sentence by sentence. Identify any statements that are NOT directly and explicitly supported by the 'Context Documents'.

            Output your analysis ONLY in the following JSON format:
            {{
              "is_grounded": <boolean, true if ALL statements in the answer are supported by the context, false otherwise>,
              "unsupported_statements": [
                {{
                  "statement": "<The specific statement from the answer that is unsupported>",
                  "reason": "<Brief explanation why it's unsupported (e.g., 'Not mentioned in context', 'Contradicts context')>"
                }}
              ]
            }}
            If the answer is fully grounded, the "unsupported_statements" list should be empty. Provide ONLY the JSON object in your response.
            """

            hallucination_model = self.llm_info.get("hallucination_check_model", "gpt-4o-mini")
            print(f"  - Performing hallucination check using {hallucination_model}...")

            response = self.llm.chat.completions.create(
                model=hallucination_model,
                messages=[
                    {"role": "system", "content": "You are a meticulous fact-checker. Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Low temperature for deterministic checking
                response_format={"type": "json_object"} # Request JSON output
            )

            analysis_json = response.choices[0].message.content
            analysis = json.loads(analysis_json)

            # Validate the received JSON structure
            if isinstance(analysis, dict) and "is_grounded" in analysis and "unsupported_statements" in analysis:
                print(f"  - Hallucination check result: Grounded = {analysis.get('is_grounded')}")
                return analysis
            else:
                st.warning("Hallucination check response was not in the expected JSON format.")
                print(f"Unexpected hallucination check JSON: {analysis_json}")
                return None

        except json.JSONDecodeError:
            st.warning("Failed to parse hallucination check response as JSON.")
            print(f"Failed to decode hallucination check JSON: {analysis_json}")
            return None
        except Exception as e:
            st.warning(f"Hallucination check encountered an error: {e}")
            print(f"Hallucination check error:")
            traceback.print_exc()
            return None


    def generate_response(self, query: str, context: Optional[Dict], temperature: float = 0.7, check_for_hallucinations: bool = True) -> Optional[Union[Dict, str]]: # Return type updated
        """
        Generates a structured answer (Dict) based on the provided context,
        or a fallback string if context is missing or errors occur.
        Includes source citation mapping and optional hallucination check.
        """
        if not self.llm:
            return "Error: The AI model (LLM) is currently unavailable."

        if not context or not context.get("documents") or not context["documents"][0]:
            # Handle cases where search returned no results gracefully
            return "I couldn't find any relevant information in the provided documents to answer your question."

        raw_docs = context["documents"][0]
        raw_metas = context.get("metadatas", [[]])[0] # Safely get metadatas

        try:
            # --- Prepare Context and Source Mapping ---
            sources_text_list = []
            source_mapping = {} # Maps citation label (e.g., "[Source 1]") to actual filename

            # Ensure metadata list length matches document list length for safe zipping
            if len(raw_docs) == len(raw_metas):
                for i, (doc, meta) in enumerate(zip(raw_docs, raw_metas)):
                    # Use filename from metadata if available, otherwise use a generic name
                    source_name = meta.get('source', f'Document Chunk {i+1}') if meta else f'Document Chunk {i+1}'
                    citation_label = f"Source {i+1}" # Use simple numeric citation labels
                    sources_text_list.append(f"[{citation_label}]\n{doc}") # Format for LLM prompt
                    source_mapping[citation_label] = source_name # Store mapping for later display
            else:
                # Fallback if metadata alignment is off (should ideally not happen)
                st.warning("Mismatch between number of documents and metadatas in context. Using generic source names.")
                for i, doc in enumerate(raw_docs):
                     citation_label = f"Source {i+1}"
                     sources_text_list.append(f"[{citation_label}]\n{doc}")
                     source_mapping[citation_label] = f"Document Chunk {i+1}" # Fallback name

            # Combine source texts into a single string for the prompt
            formatted_context = "\n\n---\n\n".join(sources_text_list)

            # --- Define the Prompt for the LLM ---
            prompt = f"""You are an assistant tasked with answering a user's question based *only* on the provided text excerpts. Do not use any external knowledge or make assumptions beyond what is written in the excerpts.

            User's Question:
            {query}

            Provided Excerpts:
            {formatted_context}

            ---
            Instructions:
            1. Analyze the excerpts carefully to understand the information relevant to the user's question.
            2. Formulate a response that directly addresses the question.
            3. Base your entire answer STRICTLY on the content of the provided excerpts.
            4. Cite the relevant source number(s) (e.g., [Source 1], [Source 2], [Source 1, Source 3]) directly after the information derived from them within the 'detailed_explanation' and 'key_points'.
            5. Structure your response ONLY as a JSON object containing the following exact keys:
               - "direct_answer": A concise, direct answer to the question (1-2 sentences).
               - "detailed_explanation": A comprehensive explanation expanding on the direct answer, synthesizing information from the excerpts and including citations like [Source X].
               - "key_points": A list of strings. Each string should represent a key takeaway or fact related to the answer, including citations like [Source X].

            Example JSON Output Format:
            {{
              "direct_answer": "The project's goal is to improve efficiency by 15% [Source 2].",
              "detailed_explanation": "The document outlines a plan to enhance operational efficiency [Source 1]. Specifically, [Source 2] mentions a target increase of 15% through process optimization. This involves streamlining workflows as detailed in [Source 1] and implementing new technology mentioned in [Source 3].",
              "key_points": [
                "The main objective is increased efficiency [Source 1].",
                "A specific target of 15% improvement is set [Source 2].",
                "Methods include process optimization and new technology [Source 1, Source 3]."
              ]
            }}

            Generate the JSON response now based *only* on the provided excerpts:
            """

            # --- Call the LLM ---
            qa_model = self.llm_info.get("qa_model", "gpt-4o-mini")
            print(f"Generating response using {qa_model} with temp={temperature}...")
            response = self.llm.chat.completions.create(
                model=qa_model,
                messages=[
                    {"role": "system", "content": "You are an expert assistant analyzing text excerpts to answer questions accurately, citing sources, and outputting ONLY in a specific JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"} # Enforce JSON output
            )

            raw_response_content = response.choices[0].message.content.strip()

            # --- Parse and Validate the LLM JSON Response ---
            try:
                structured_answer = json.loads(raw_response_content)

                # Basic validation of the structure
                required_keys = ["direct_answer", "detailed_explanation", "key_points"]
                if not isinstance(structured_answer, dict) or not all(k in structured_answer for k in required_keys):
                    st.warning("LLM response was valid JSON but missing required keys. Displaying raw response.")
                    print(f"Warning: LLM JSON response missing keys. Raw response: {raw_response_content}")
                    # Fallback to returning the raw JSON string for debugging
                    return f"LLM Response (unexpected format):\n```json\n{raw_response_content}\n```"

                # Add the source mapping to the result dictionary for UI use
                structured_answer["source_mapping"] = source_mapping
                structured_answer["hallucination_warning"] = None # Initialize warning field

                # --- Optional: Hallucination Check ---
                if check_for_hallucinations:
                    # Check the most detailed part of the answer for grounding
                    text_to_check = structured_answer.get("detailed_explanation", "")
                    if text_to_check:
                        hallucination_result = self._check_hallucination(query, raw_docs, text_to_check)
                        if hallucination_result and not hallucination_result.get("is_grounded", True):
                            warnings_list = hallucination_result.get("unsupported_statements", [])
                            warning_messages = [f"- \"{s.get('statement','N/A')}\" ({s.get('reason','N/A')})" for s in warnings_list]
                            warning_text = f"**⚠️ Potential Hallucination Warning:** The AI's explanation might contain statements not fully supported by the source documents:\n" + "\n".join(warning_messages)
                            structured_answer["hallucination_warning"] = warning_text # Add warning to the dictionary

                return structured_answer # Return the validated and potentially annotated dictionary

            except json.JSONDecodeError as json_e:
                st.error(f"Failed to parse the AI's response as JSON: {json_e}")
                print(f"JSONDecodeError for LLM response. Raw content was:\n{raw_response_content}")
                # Fallback: return the raw response string if JSON parsing fails
                return f"Error: Couldn't understand the AI's structured answer. Raw response:\n```\n{raw_response_content}\n```"
            except Exception as parse_e:
                st.error(f"An error occurred while processing the AI's response: {parse_e}")
                print(f"Error processing LLM response:")
                traceback.print_exc()
                return f"Error processing AI response: {parse_e}"

        except Exception as e:
            st.error(f"An unexpected error occurred during response generation: {e}")
            print(f"Generate response error:")
            traceback.print_exc()
            return f"An error occurred while generating the response: {e}" # Return error string


    def generate_document_summary(self, document_text: str, document_name: str) -> Optional[str]:
        """
        Uses an LLM to generate a brief summary of the provided document text.
        Returns the summary string or None on error.
        """
        if not self.llm:
            st.warning(f"Summary generation skipped for '{document_name}': LLM client unavailable.")
            return None
        if not document_text:
            st.warning(f"Summary generation skipped for '{document_name}': No text content provided.")
            return None

        try:
            summary_model = self.llm_info.get("summary_model", "gpt-4o-mini")
            print(f"Generating summary for '{document_name}' using {summary_model}...")

            # Truncate input text if it's too long to avoid excessive API cost/time
            truncated_text = document_text
            if len(document_text) > MAX_SUMMARY_INPUT_CHARS:
                truncated_text = document_text[:MAX_SUMMARY_INPUT_CHARS] + "... (document truncated for summary)"
                print(f"  - Document text truncated to {MAX_SUMMARY_INPUT_CHARS} chars for summary generation.")

            prompt = f"""Please provide a very brief (2-3 sentence) summary of the main topics or purpose of the following document content. Focus on the core subject matter.

            Document Content:
            \"\"\"
            {truncated_text}
            \"\"\"

            Brief Summary (2-3 sentences):
            """

            response = self.llm.chat.completions.create(
                model=summary_model,
                messages=[
                    {"role": "system", "content": "You are an assistant that writes brief, concise summaries of documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, # Low temperature for factual summary
                max_tokens=150, # Limit output length
                n=1,
                stop=None,
            )

            summary = response.choices[0].message.content.strip()
            print(f"  - Summary generated successfully for '{document_name}'.")
            return summary

        except Exception as e:
            st.error(f"Failed to generate summary for '{document_name}': {e}")
            print(f"Error generating summary for {document_name}:")
            traceback.print_exc()
            return None # Return None indicating failure


    def get_embedding_info(self) -> Dict[str, Any]:
        """Returns details about the currently used embedding model."""
        if not self.emb_info: return {} # Should not happen if constructor succeeded
        return {
            "name": self.emb_info.get("name", "Unknown"),
            "dimensions": self.emb_info.get("dimensions", "N/A"),
            "model_provider": self.embedding_model_provider,
            "model_name": self.emb_info.get("model_name", "N/A")
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the Chroma collection (chunk count, unique sources)."""
        if not self.collection:
            return {"count": 0, "sources": []}
        try:
            count = self.collection.count()
            # Fetch sources efficiently - get metadata only, maybe batched if needed
            # Limit fetch if collection is massive to avoid performance hit
            limit = 10000
            items = self.collection.get(limit=limit, include=['metadatas'])
            sources = set()
            if items and items.get("metadatas"):
                sources = {m["source"] for m in items["metadatas"] if m and "source" in m}

            # Add a note if the source list might be incomplete due to the limit
            source_note = ""
            if count > limit and len(sources) < count: # Heuristic check
                 source_note = f"(from first {limit} chunks)"

            return {"count": count, "sources": sorted(list(sources)), "source_note": source_note}
        except Exception as e:
            st.error(f"Error getting collection stats: {e}")
            print(f"Get collection stats error:")
            traceback.print_exc()
            return {"count": 0, "sources": [], "source_note": "(Error retrieving stats)"}

    def delete_document_by_source(self, source_name: str) -> bool:
        """Deletes all chunks associated with a specific source filename and updates indexes."""
        if not source_name:
            st.warning("No document source name provided for deletion.")
            return False
        if not self.collection:
            st.error("Cannot delete document: Collection is not available.")
            return False

        try:
            st.info(f"Finding chunks associated with source: '{source_name}'...")
            # Find IDs of chunks matching the source metadata
            # Note: This might be slow on very large collections without metadata indexing.
            # ChromaDB is improving metadata filtering performance.
            results = self.collection.get(where={"source": source_name}, include=[]) # Only need IDs
            ids_to_delete = results.get("ids")

            if not ids_to_delete:
                st.warning(f"No document chunks found with source name '{source_name}'. Nothing to delete.")
                return False # Or True, as the desired state (no docs with that name) is achieved? Let's say False as no action was taken.

            st.info(f"Found {len(ids_to_delete)} chunks. Deleting document '{source_name}' from the knowledge base...")
            # Delete the chunks by their IDs
            self.collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks for source '{source_name}'.")

            # Crucially, update the keyword search index after deletion
            print("Updating keyword index after deletion...")
            time.sleep(0.5) # Short pause potentially helpful for DB consistency before re-indexing
            self._update_keyword_search_index_from_db()

            # Let the Streamlit UI handle the success message after rerun
            return True

        except Exception as e:
            st.error(f"An error occurred while deleting document '{source_name}': {e}")
            print(f"Delete document by source error:")
            traceback.print_exc()
            return False


    def reset_collection(self) -> bool:
        """Deletes the entire ChromaDB collection and resets the keyword index."""
        if not self.collection:
            st.info("Collection does not exist or is not initialized. No reset needed.")
            return True # Already in the desired state

        try:
            collection_name_to_delete = self.collection.name
            st.warning(f"Resetting entire knowledge base (collection: '{collection_name_to_delete}')...")
            count = self.collection.count()
            if count == 0:
                st.info("Collection is already empty.")
                # Still proceed to delete and recreate to ensure clean state if needed
            else:
                 st.info(f"Deleting {count} items from collection '{collection_name_to_delete}'...")

            self.db.delete_collection(name=collection_name_to_delete)
            print(f"Collection '{collection_name_to_delete}' deleted.")

            # Re-initialize the collection
            st.info("Re-initializing collection...")
            self.collection = self._setup_collection() # This will create a new, empty collection

            # Reset keyword search components
            print("Resetting keyword index...")
            self.corpus = []
            self.doc_ids = []
            self.doc_metadatas = []
            self.bm25 = None
            # No need to call _update_keyword_search_index_from_db() as it's empty

            st.success(f"Knowledge base '{collection_name_to_delete}' has been reset.")
            return True

        except Exception as e:
            st.error(f"Failed to reset knowledge base: {e}")
            print(f"Reset collection error:")
            traceback.print_exc()
            return False

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="SmartBot Doc Q&A", page_icon="📚", layout="wide")
    st.title("📚 SmartBot Doc Q&A")
    st.caption("Upload Documents (PDF, DOCX, TXT), ask questions, and get answers grounded in your data.")

    # --- Session State Initialization ---
    # Ensure keys exist before accessing them
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "upload_key_counter" not in st.session_state:
        st.session_state.upload_key_counter = 0 # To reset file uploader
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Stores {"role": "user/assistant", "content": "..."}
    if "confirming_delete" not in st.session_state:
        st.session_state.confirming_delete = None # Stores filename for delete confirmation
    if "upload_success_message" not in st.session_state:
        st.session_state.upload_success_message = None # Feedback after upload
    # NEW: Store document summaries in session state
    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = {} # Dict: {filename: summary_string}

    # --- RAG System Initialization ---
    # Initialize only once per session
    try:
        if st.session_state.rag_system is None:
            with st.spinner("Initializing Knowledge Base Connection... Please wait."):
                model_selector_init = TheModelSelector()
                llm_provider, embedding_provider = model_selector_init.get_models()
                st.session_state.rag_system = TheRAGSystem(
                    embedding_model_provider=embedding_provider,
                    llm_provider=llm_provider
                )
            # No explicit rerun needed here, sidebar/tabs will use the initialized system
        rag_system: TheRAGSystem = st.session_state.rag_system # Assign for easier access, with type hint

        # --- Sidebar Information ---
        with st.sidebar:
            st.header("📊 Knowledge Base Status")
            stats = rag_system.get_collection_stats()
            st.metric("Total Indexed Chunks", stats.get("count", 0))
            st.metric("Indexed Documents", len(stats.get("sources", [])))
            if stats.get("source_note"):
                st.caption(stats["source_note"])
            st.divider()
            st.header("⚙️ Configuration")
            model_selector_disp = TheModelSelector() # Re-instantiate for display if needed
            llm_disp_info = model_selector_disp.get_llm_info()
            emb_disp_info = rag_system.get_embedding_info() # Get from initialized system
            st.info(f"**LLM:** {llm_disp_info['qa_model']} (for Q&A)\n"
                    f"**Summarizer:** {llm_disp_info['summary_model']}\n"
                    f"**Embeddings:** {emb_disp_info['name']}\n"
                    f" (Provider: {emb_disp_info['model_provider']}, Dim: {emb_disp_info['dimensions']})")
            st.info("**Retrieval:** Hybrid Search (RRF)")
            st.divider()
            # Optional: Add Reset Button Here? Or keep in Manage Tab?
            st.warning("Resetting the base deletes all data.")
            if st.button("⚠️ Reset Entire Knowledge Base", key="sidebar_reset_button"):
                st.session_state.confirming_reset = True # Use state to trigger confirmation modal/logic

            # Confirmation logic for Reset
            if st.session_state.get("confirming_reset", False):
                 st.error("Are you sure you want to delete ALL indexed data? This cannot be undone.")
                 c1, c2 = st.columns(2)
                 with c1:
                    if st.button("Confirm Reset", type="primary", key="sidebar_confirm_reset"):
                        with st.spinner("Resetting knowledge base..."):
                             reset_success = rag_system.reset_collection()
                             # Also clear summaries from session state
                             st.session_state.document_summaries = {}
                             st.session_state.confirming_reset = False # Clear flag
                             if reset_success:
                                 st.success("Knowledge base reset successfully.")
                                 time.sleep(1.5)
                             else:
                                 st.error("Failed to reset knowledge base.")
                                 time.sleep(1.5)
                             st.rerun() # Rerun to reflect changes
                 with c2:
                    if st.button("Cancel Reset", key="sidebar_cancel_reset"):
                         st.session_state.confirming_reset = False # Clear flag
                         st.rerun()

    except Exception as e:
        st.error(f"Fatal error during RAG system initialization: {e}")
        print("Fatal RAG Initialization Error:")
        traceback.print_exc()
        st.stop() # Stop Streamlit app execution if RAG system fails critically

    # --- Define Tabs ---
    tab_chat, tab_upload, tab_view = st.tabs(["💬 Chat", "➕ Upload Documents", "📄 Manage Documents"])

    # ==========================
    # ---     Chat Tab       ---
    # ==========================
    with tab_chat:
        st.header("Ask Questions About Your Documents")
        current_stats = rag_system.get_collection_stats() # Get current stats for this tab

        if current_stats["count"] == 0:
            st.info("The knowledge base is empty. Please upload documents in the 'Upload Documents' tab first.")
        else:
            # --- Chat Controls ---
            col_context, col_temp = st.columns([3, 1])
            with col_context:
                # Create list of available sources for context filtering
                available_sources = ["All Documents"] + current_stats.get("sources", [])
                selected_context = st.selectbox(
                    "Limit context to specific document (optional):",
                    options=available_sources,
                    key="chat_context_select",
                    help="Choose 'All Documents' to search across everything, or select a specific file to focus the search."
                )
            with col_temp:
                temperature = st.slider(
                    "LLM Temperature (Creativity)", 0.0, 1.0, 0.5, 0.05, # Adjusted default and step
                    key="chat_temp_slider",
                    help="Lower values (e.g., 0.1) produce more focused, deterministic answers. Higher values (e.g., 0.9) allow for more creativity and variation."
                )

            # Determine the filter for the RAG query based on selection
            query_filter = {"source": selected_context} if selected_context != "All Documents" else None
            if query_filter:
                st.caption(f"ℹ️ Answers will be based primarily on content from: **{selected_context}**")

            # --- Chat History Display ---
            st.markdown("---")
            st.subheader("Conversation")
            if not st.session_state.chat_history:
                st.caption("No questions asked yet in this session.")

            # Display previous messages from session state
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"]) # Display pre-formatted markdown content

            # --- Chat Input and Processing ---
            user_query = st.chat_input("Enter your question here...")

            if user_query:
                # Add user query to history and display it
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Process query and generate assistant response
                with st.chat_message("assistant"):
                    # Use a placeholder for streaming-like effect
                    placeholder = st.empty()
                    placeholder.markdown("Thinking... 🤔")
                    formatted_response_content = "" # Initialize formatted response string

                    try:
                        start_time = time.time()

                        # 1. Retrieve relevant context documents
                        context = rag_system.query_documents(
                            query=user_query,
                            n_results=5, # Retrieve slightly more context for potentially better answers
                            where_filter=query_filter
                        )

                        # 2. Generate response using the context
                        response_object = rag_system.generate_response(
                            query=user_query,
                            context=context,
                            temperature=temperature,
                            check_for_hallucinations=True # Enable hallucination check
                        )
                        end_time = time.time()
                        print(f"Chat query processing time: {end_time - start_time:.2f}s")

                        # --- Format the Response for Display ---
                        if isinstance(response_object, dict):
                            # It's the structured dictionary - format it nicely
                            display_parts = []

                            # Add Hallucination Warning (if present)
                            if response_object.get("hallucination_warning"):
                                display_parts.append(response_object["hallucination_warning"]) # Should be pre-formatted markdown

                            # Add Direct Answer
                            if response_object.get("direct_answer"):
                                display_parts.append(f"**Answer:** {response_object['direct_answer']}")

                            # Add Detailed Explanation
                            if response_object.get("detailed_explanation"):
                                prefix = "\n---\n" if display_parts else "" # Separator
                                display_parts.append(f"{prefix}**Explanation:**\n{response_object['detailed_explanation']}")

                            # Add Key Points
                            if response_object.get("key_points"):
                                points_markdown = "\n".join([f"- {point}" for point in response_object["key_points"]])
                                prefix = "\n---\n" if display_parts else "" # Separator
                                display_parts.append(f"{prefix}**Key Points:**\n{points_markdown}")

                            # Combine parts into the final markdown string
                            formatted_response_content = "\n\n".join(display_parts)

                            # --- Add Expander for Raw Sources ---
                            sources = context.get("documents", [[]])[0] if context else []
                            metadatas = context.get("metadatas", [[]])[0] if context else []
                            source_mapping = response_object.get("source_mapping", {}) # Get mapping from response dict

                            if sources and metadatas and source_mapping:
                                with st.expander("View retrieved source document snippets", expanded=False):
                                    # Create a reverse map: filename -> citation label for display
                                    citation_label_map = {name: label for label, name in source_mapping.items()}

                                    for idx, (doc_text, meta) in enumerate(zip(sources, metadatas)):
                                        source_file = meta.get("source", "Unknown Source")
                                        # Use the mapped citation label if available, else fallback
                                        citation_label = citation_label_map.get(source_file, f"Snippet {idx+1}")
                                        st.markdown(f"**[{citation_label}] `{source_file}`** (Chars: {meta.get('start', '?')}-{meta.get('end', '?')})")
                                        # Display a snippet of the source text
                                        display_doc = doc_text[:600] + "..." if len(doc_text) > 600 else doc_text
                                        st.text_area(
                                            f"Source Snippet {idx}", display_doc, height=120,
                                            disabled=True, label_visibility="collapsed", key=f"chat_src_{idx}_{user_query[:10]}" # More unique key
                                        )
                                    st.caption("Note: These are the raw text snippets provided to the AI for generating the answer above.")

                        elif isinstance(response_object, str):
                            # It's already a formatted string (e.g., error message, fallback)
                            formatted_response_content = response_object
                        else:
                            # Handle unexpected response types gracefully
                            formatted_response_content = "Sorry, I received an unexpected response format from the AI."
                            print(f"Unexpected response type in chat: {type(response_object)}")

                        # Display the final formatted response
                        placeholder.markdown(formatted_response_content)

                    except Exception as chat_e:
                        # Catchall for errors during the chat generation process
                        formatted_response_content = f"An error occurred while processing your request: {chat_e}"
                        placeholder.error(formatted_response_content)
                        print(f"Chat processing error:")
                        traceback.print_exc()

                # Add the final formatted assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": formatted_response_content})

            # --- Clear Chat Button ---
            st.markdown("---")
            if st.button("Clear Chat History", key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.rerun() # Rerun to clear the display

    # ==========================
    # ---   Upload Tab       ---
    # ==========================
    with tab_upload:
        st.header("Upload New Document")
        st.markdown("Add PDF, DOCX, or TXT files to the knowledge base. The system will process the text, generate a brief summary, and index the content for Q&A.")

        # --- Advanced Options ---
        with st.expander("Advanced Processing Options"):
            chunk_size_opt = st.slider(
                "Target Chunk Size (characters)", 300, 2000, CHUNK_SIZE, 100,
                key="upload_chunk_size",
                help="Approximate size of text chunks indexed. Smaller chunks offer more precise retrieval but less context; larger chunks provide more context but might be less specific."
            )
            chunk_overlap_opt = st.slider(
                "Chunk Overlap (characters)", 0, 500, CHUNK_OVERLAP, 50,
                key="upload_chunk_overlap",
                help="Number of characters shared between consecutive chunks to maintain context continuity."
            )

        # --- File Uploader ---
        # Use the counter in the key to allow re-uploading the same file after processing
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False, # Process one file at a time for clarity
            key=f"file_uploader_{st.session_state.upload_key_counter}"
        )

        if uploaded_file is not None:
            st.markdown("---")
            file_details_cols = st.columns(2)
            with file_details_cols[0]:
                st.write(f"**Selected File:**")
                st.write(f"`{uploaded_file.name}`")
            with file_details_cols[1]:
                st.write(f"**File Size:**")
                st.write(f"{uploaded_file.size / 1024:.1f} KB")


            # Check if a document with the same name already exists
            existing_sources = rag_system.get_collection_stats().get("sources", [])
            process_button_label = "Process & Add Document"
            is_existing = uploaded_file.name in existing_sources
            if is_existing:
                st.warning(f"⚠️ A document named **'{uploaded_file.name}'** already exists in the knowledge base. Processing again will add its content anew (potentially creating duplicates if the content is identical). You can delete the existing document first from the 'Manage Documents' tab if you want to replace it.")
                process_button_label = "Process & Add Anyway"

            # --- Process Button ---
            if st.button(process_button_label, type="primary", key=f"process_btn_{st.session_state.upload_key_counter}"):
                overall_start_time = time.time()
                with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment. Reading file..."):
                    doc_processed_successfully = False
                    try:
                        # 1. Read Document Text
                        processor = TheDocProcessor(chunk_size=chunk_size_opt, chunk_overlap=chunk_overlap_opt)
                        read_start = time.time()
                        document_text = processor.read_document(uploaded_file, uploaded_file.name)
                        read_time = time.time() - read_start
                        if not document_text:
                            st.error("Failed to extract any text from the document. Cannot proceed.")
                            # No need to stop explicitly, just won't proceed
                        else:
                            st.info(f"Read document ({len(document_text):,} chars) in {read_time:.2f}s.")

                            # 2. Generate Summary (NEW STEP)
                            st.spinner(f"Processing '{uploaded_file.name}'... Generating summary...")
                            summary_start = time.time()
                            generated_summary = rag_system.generate_document_summary(document_text, uploaded_file.name)
                            summary_time = time.time() - summary_start
                            if generated_summary:
                                st.info(f"Generated summary in {summary_time:.2f}s.")
                                # Store the summary in session state immediately
                                st.session_state.document_summaries[uploaded_file.name] = generated_summary
                            else:
                                st.warning("Could not generate summary for this document.")
                                st.session_state.document_summaries[uploaded_file.name] = "Summary generation failed." # Store placeholder

                            # 3. Create Chunks
                            st.spinner(f"Processing '{uploaded_file.name}'... Splitting into chunks...")
                            chunking_start = time.time()
                            chunks = processor.create_chunks(document_text, uploaded_file)
                            chunking_time = time.time() - chunking_start
                            if not chunks:
                                st.error("Failed to split the document into chunks. Cannot add to knowledge base.")
                            else:
                                st.info(f"Created {len(chunks)} chunks in {chunking_time:.2f}s.")

                                # 4. Add Chunks to RAG System (Vector DB + BM25)
                                st.spinner(f"Processing '{uploaded_file.name}'... Adding chunks to knowledge base...")
                                add_start = time.time()
                                success = rag_system.add_documents(chunks)
                                add_time = time.time() - add_start

                                if success:
                                    total_time = time.time() - overall_start_time
                                    st.session_state.upload_success_message = (
                                        f"✅ Successfully processed and added **'{uploaded_file.name}'** "
                                        f"({len(chunks)} chunks) to the knowledge base in {total_time:.2f} seconds."
                                    )
                                    doc_processed_successfully = True
                                else:
                                    st.error(f"❌ Failed to add document chunks for '{uploaded_file.name}' to the database.")
                                    # Clean up summary if adding failed? Or keep it? Let's keep it for now.
                                    # if uploaded_file.name in st.session_state.document_summaries:
                                    #     del st.session_state.document_summaries[uploaded_file.name]


                    except Exception as upload_e:
                        st.error(f"An unexpected error occurred during processing: {upload_e}")
                        print(f"Upload processing error for {uploaded_file.name}:")
                        traceback.print_exc()
                        # Clean up summary if processing failed badly
                        if uploaded_file.name in st.session_state.document_summaries:
                            del st.session_state.document_summaries[uploaded_file.name]

                    finally:
                        # Increment key and rerun ONLY if processing was fully successful
                        # This clears the file uploader and displays the success message
                        if doc_processed_successfully:
                             st.session_state.upload_key_counter += 1
                             st.rerun()


        # Display success message from session state (if it exists after a rerun)
        if st.session_state.upload_success_message:
            st.success(st.session_state.upload_success_message)
            st.session_state.upload_success_message = None # Clear the message after displaying it once


    # ==========================
    # --- Manage Documents Tab ---
    # ==========================
    with tab_view:
        st.header("View and Manage Uploaded Documents")
        view_stats = rag_system.get_collection_stats()
        view_sources = view_stats.get("sources", [])

        if not view_sources:
            st.info("No documents have been uploaded yet. Use the '➕ Upload Documents' tab to add files.")
        else:
            st.markdown(f"There are **{len(view_sources)}** document(s) indexed in the knowledge base, comprising **{view_stats.get('count', 0)}** text chunks.")
            st.markdown("---")

            # --- Display Indexed Documents with Summaries ---
            st.subheader("📚 Indexed Documents")
            if not st.session_state.document_summaries:
                 st.caption("Document summaries might still be loading or were not generated.")

            for i, src_name in enumerate(view_sources, 1):
                 with st.container(): # Group document name and summary
                    st.markdown(f"**{i}. `{src_name}`**")
                    # Retrieve and display the summary from session state
                    summary = st.session_state.document_summaries.get(src_name, "_Summary not available or not generated yet._")
                    st.caption(f"Summary: {summary}")
                    st.divider()


            # --- Delete Document Section ---
            st.subheader("🗑️ Delete Document")
            st.markdown("Select a document from the list below to remove it and all its associated data from the knowledge base.")

            # Create options list including a blank default
            delete_options = [""] + view_sources
            doc_to_delete = st.selectbox(
                 "Select document to remove:",
                 options=delete_options,
                 index=0, # Default to blank
                 key="view_doc_delete_select",
                 help="Choosing a document enables the 'Delete' button."
            )

            # Enable delete button only if a document is selected
            delete_disabled = not bool(doc_to_delete)
            delete_btn_key = f"view_delete_doc_btn_{doc_to_delete}" if doc_to_delete else "view_delete_doc_btn_disabled"

            if st.button("Delete Selected Document", type="secondary", disabled=delete_disabled, key=delete_btn_key):
                 if doc_to_delete:
                      # Set confirmation flag in session state, triggering the confirmation dialog on rerun
                      st.session_state.confirming_delete = doc_to_delete
                      st.rerun() # Rerun immediately to show the confirmation

            elif delete_disabled:
                 st.caption("Select a document above to enable deletion.")

            # --- Confirmation Dialog Logic (triggered by confirming_delete state) ---
            if st.session_state.confirming_delete:
                 doc_name_to_confirm = st.session_state.confirming_delete
                 st.error(f"**Confirm Deletion:** Are you sure you want to permanently remove **'{doc_name_to_confirm}'** and all its data? This action cannot be undone.")
                 confirm_col, cancel_col = st.columns(2)
                 with confirm_col:
                     if st.button(f"Yes, Delete '{doc_name_to_confirm}'", type="primary", key=f"confirm_del_{doc_name_to_confirm}"):
                         with st.spinner(f"Deleting '{doc_name_to_confirm}'..."):
                             delete_success = rag_system.delete_document_by_source(doc_name_to_confirm)
                             # Also remove the summary from session state
                             st.session_state.document_summaries.pop(doc_name_to_confirm, None) # Safely remove

                         st.session_state.confirming_delete = None # Clear confirmation state regardless of success
                         if delete_success:
                             st.success(f"Successfully deleted '{doc_name_to_confirm}'.")
                             time.sleep(1.5) # Brief pause to show message
                         else:
                             st.error(f"Failed to delete '{doc_name_to_confirm}'. Check system logs for details.")
                             time.sleep(2.0) # Longer pause for error
                         st.rerun() # Rerun to update the document list and clear dialog

                 with cancel_col:
                     if st.button("Cancel Deletion", key=f"cancel_del_{doc_name_to_confirm}"):
                          st.session_state.confirming_delete = None # Clear confirmation state
                          st.rerun() # Rerun to hide the confirmation dialog


if __name__ == "__main__":
    main()