from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv

# Import your existing RAG system classes
from rag_query_system import TheModelSelector, ThePDFProcessor, TheRAGSystem

# Load environment variables
load_dotenv()

app = FastAPI(
    title="RAG System API",
    description="API for retrieving answers from documents using RAG",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the RAG system
def get_rag_system():
    model_selector = TheModelSelector()
    llm_model, embedding_model = model_selector.get_models()
    rag_system = TheRAGSystem(embedding_model, llm_model)
    return rag_system

# Request and response models
class QueryRequest(BaseModel):
    query: str
    n_results: int = 3
    temperature: float = 0.7

class SourceInfo(BaseModel):
    source: str
    text: str

class QueryResponse(BaseModel):
    direct_answer: str
    detailed_explanation: str
    key_points: List[str]
    sources: List[SourceInfo]

class DocumentListResponse(BaseModel):
    count: int
    sources: List[str]

@app.get("/")
async def root():
    return {"message": "RAG System API is running. See /docs for API documentation."}

@app.get("/documents", response_model=DocumentListResponse)
async def get_documents(rag_system: TheRAGSystem = Depends(get_rag_system)):
    """
    Get a list of all documents in the system
    """
    stats = rag_system.get_collection_stats()
    return {"count": stats["count"], "sources": stats["sources"]}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, rag_system: TheRAGSystem = Depends(get_rag_system)):
    """
    Query the RAG system with a natural language question
    """
    # Find relevant document pieces
    results = rag_system.query_documents(request.query, n_results=request.n_results)
    
    if not results or not results["documents"] or not results["documents"][0]:
        raise HTTPException(status_code=404, detail="No relevant information found in the documents")
    
    # Generate the answer
    response = rag_system.generate_response(
        request.query, results["documents"][0], temperature=request.temperature
    )
    
    if not response:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
    # Parse the response
    try:
        # Extract direct answer
        if "DIRECT ANSWER:" in response:
            direct_answer = response.split("DIRECT ANSWER:")[1].split("DETAILED EXPLANATION:")[0].strip()
        else:
            direct_answer = response.split("\n\n")[0].strip()
            
        # Extract detailed explanation
        if "DETAILED EXPLANATION:" in response and "KEY POINTS:" in response:
            detailed_explanation = response.split("DETAILED EXPLANATION:")[1].split("KEY POINTS:")[0].strip()
        elif "DETAILED EXPLANATION:" in response:
            detailed_explanation = response.split("DETAILED EXPLANATION:")[1].strip()
        else:
            parts = response.split("\n\n")
            detailed_explanation = "\n\n".join(parts[1:-1]) if len(parts) > 2 else response
            
        # Extract key points
        key_points = []
        if "KEY POINTS:" in response:
            key_points_text = response.split("KEY POINTS:")[1].strip()
            for point in key_points_text.split("\n"):
                if point.strip() and point.strip() != "-":
                    key_points.append(point.strip().replace("- ", ""))
        
        # Prepare source info
        sources = []
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            source = metadata.get("source", f"Source {i+1}")
            sources.append({"source": source, "text": doc})
        
        return {
            "direct_answer": direct_answer,
            "detailed_explanation": detailed_explanation,
            "key_points": key_points,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")

@app.get("/info")
async def get_model_info(rag_system: TheRAGSystem = Depends(get_rag_system)):
    """
    Get information about the models being used
    """
    embedding_info = rag_system.get_embedding_info()
    return {
        "embedding_model": embedding_info["name"],
        "dimensions": embedding_info["dimensions"],
        "llm_model": "GPT-4o-mini"
    }

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
