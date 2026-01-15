from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
from pathlib import Path

from services.rag_engine import RAGEngine
from services.code_generator import CodeGenerator
from services.data_processor import DataProcessor

app = FastAPI(title="Ultimate RAG Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()
code_generator = CodeGenerator()
data_processor = DataProcessor()

class RAGConfig(BaseModel):
    rag_type: str  # vanilla, agentic, graph, hybrid
    model_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "openai"
    vector_db: str = "chromadb"
    
class QueryRequest(BaseModel):
    query: str
    session_id: str
    top_k: int = 5

class DatabaseConfig(BaseModel):
    db_type: str  # postgres, mysql, mongodb
    connection_string: str
    tables: Optional[List[str]] = None

@app.post("/upload")
async def upload_data(
    files: List[UploadFile] = File(...),
    rag_type: str = Form(...),
    model_name: str = Form(...),
    chunk_size: int = Form(1000),
    embedding_model: str = Form("openai")
):
    """Upload any format: PDF, DOCX, TXT, CSV, JSON, Images, Audio, Video"""
    try:
        session_id = data_processor.create_session()
        processed_files = []
        
        for file in files:
            file_path = await data_processor.save_file(file, session_id)
            processed_data = await data_processor.process_file(file_path)
            processed_files.append(processed_data)
        
        config = RAGConfig(
            rag_type=rag_type,
            model_name=model_name,
            chunk_size=chunk_size,
            embedding_model=embedding_model
        )
        
        rag_instance = await rag_engine.create_rag(session_id, config, processed_files)
        
        return {
            "session_id": session_id,
            "status": "success",
            "files_processed": len(files),
            "rag_type": rag_type,
            "message": "RAG system created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/connect-database")
async def connect_database(config: DatabaseConfig):
    """Connect to database and create RAG from tables"""
    try:
        session_id = data_processor.create_session()
        db_data = await data_processor.extract_from_database(config)
        
        rag_config = RAGConfig(
            rag_type="vanilla",
            model_name="gpt-4",
            embedding_model="openai"
        )
        
        rag_instance = await rag_engine.create_rag(session_id, rag_config, db_data)
        
        return {
            "session_id": session_id,
            "status": "success",
            "tables_processed": len(config.tables) if config.tables else "all",
            "message": "Database RAG created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    try:
        response = await rag_engine.query(
            request.session_id,
            request.query,
            request.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-code")
async def generate_code(
    session_id: str = Form(...),
    language: str = Form("python"),
    include_frontend: bool = Form(False)
):
    """Generate complete RAG code based on configuration"""
    try:
        rag_config = rag_engine.get_config(session_id)
        code = code_generator.generate(rag_config, language, include_frontend)
        
        zip_path = code_generator.create_project_zip(session_id, code)
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"rag_project_{session_id}.zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag-types")
async def get_rag_types():
    """Get available RAG types with descriptions"""
    return {
        "vanilla": {
            "name": "Vanilla RAG",
            "description": "Simple retrieval-augmented generation",
            "use_case": "Basic Q&A, documentation search",
            "complexity": "Low",
            "category": "Basic"
        },
        "agentic": {
            "name": "Agentic RAG",
            "description": "Multi-agent system with reasoning",
            "use_case": "Complex queries, multi-step reasoning",
            "complexity": "High",
            "category": "Advanced"
        },
        "graph": {
            "name": "Graph RAG",
            "description": "Knowledge graph-based retrieval",
            "use_case": "Relationship queries, connected data",
            "complexity": "Medium",
            "category": "Advanced"
        },
        "corrective": {
            "name": "Corrective RAG (CRAG)",
            "description": "Self-correcting retrieval with web fallback",
            "use_case": "Accuracy-critical applications",
            "complexity": "High",
            "category": "Advanced"
        },
        "self_rag": {
            "name": "Self-RAG",
            "description": "Self-reflective retrieval with quality assessment",
            "use_case": "High-quality responses, fact-checking",
            "complexity": "High",
            "category": "Advanced"
        },
        "adaptive": {
            "name": "Adaptive RAG",
            "description": "Dynamically adjusts retrieval strategy",
            "use_case": "Variable query complexity",
            "complexity": "High",
            "category": "Advanced"
        },
        "modular": {
            "name": "Modular RAG",
            "description": "Composable RAG components",
            "use_case": "Custom workflows, flexibility",
            "complexity": "Medium",
            "category": "Advanced"
        },
        "fusion": {
            "name": "RAG Fusion",
            "description": "Multiple query generation and ranking",
            "use_case": "Comprehensive search, diverse perspectives",
            "complexity": "Medium",
            "category": "Advanced"
        },
        "hyde": {
            "name": "HyDE (Hypothetical Document Embeddings)",
            "description": "Generate hypothetical answers for better retrieval",
            "use_case": "Semantic search improvement",
            "complexity": "Medium",
            "category": "Advanced"
        },
        "raptor": {
            "name": "RAPTOR",
            "description": "Recursive abstractive processing with tree organization",
            "use_case": "Long documents, hierarchical data",
            "complexity": "High",
            "category": "Advanced"
        },
        "colbert": {
            "name": "ColBERT RAG",
            "description": "Late interaction retrieval",
            "use_case": "High-precision retrieval",
            "complexity": "Medium",
            "category": "Advanced"
        },
        "parent_document": {
            "name": "Parent Document Retriever",
            "description": "Retrieve small chunks, return full context",
            "use_case": "Context preservation",
            "complexity": "Low",
            "category": "Basic"
        },
        "multi_query": {
            "name": "Multi-Query RAG",
            "description": "Generate multiple queries for comprehensive retrieval",
            "use_case": "Complex questions, multiple angles",
            "complexity": "Medium",
            "category": "Intermediate"
        },
        "contextual_compression": {
            "name": "Contextual Compression",
            "description": "Compress retrieved context to relevant parts",
            "use_case": "Token optimization, cost reduction",
            "complexity": "Medium",
            "category": "Intermediate"
        },
        "reranking": {
            "name": "Reranking RAG",
            "description": "Two-stage retrieval with reranking",
            "use_case": "Improved relevance, precision",
            "complexity": "Medium",
            "category": "Intermediate"
        },
        "ensemble": {
            "name": "Ensemble RAG",
            "description": "Multiple retrievers combined",
            "use_case": "Diverse data sources",
            "complexity": "Medium",
            "category": "Intermediate"
        },
        "sql": {
            "name": "SQL RAG",
            "description": "Natural language to SQL queries",
            "use_case": "Database querying",
            "complexity": "Medium",
            "category": "Specialized"
        },
        "multimodal": {
            "name": "Multimodal RAG",
            "description": "Text, images, audio, video retrieval",
            "use_case": "Mixed media content",
            "complexity": "High",
            "category": "Specialized"
        },
        "temporal": {
            "name": "Temporal RAG",
            "description": "Time-aware retrieval",
            "use_case": "Historical data, time-series",
            "complexity": "Medium",
            "category": "Specialized"
        },
        "conversational": {
            "name": "Conversational RAG",
            "description": "Context-aware multi-turn conversations",
            "use_case": "Chatbots, assistants",
            "complexity": "Medium",
            "category": "Specialized"
        },
        "streaming": {
            "name": "Streaming RAG",
            "description": "Real-time streaming responses",
            "use_case": "Live updates, real-time data",
            "complexity": "High",
            "category": "Specialized"
        },
        "federated": {
            "name": "Federated RAG",
            "description": "Query across multiple distributed sources",
            "use_case": "Multi-source enterprise data",
            "complexity": "High",
            "category": "Enterprise"
        },
        "hierarchical": {
            "name": "Hierarchical RAG",
            "description": "Multi-level retrieval strategy",
            "use_case": "Structured documents, taxonomies",
            "complexity": "Medium",
            "category": "Enterprise"
        },
        "semantic_cache": {
            "name": "Semantic Cache RAG",
            "description": "Cache similar queries semantically",
            "use_case": "Performance optimization",
            "complexity": "Low",
            "category": "Optimization"
        },
        "active_rag": {
            "name": "Active RAG",
            "description": "Active learning with user feedback",
            "use_case": "Continuous improvement",
            "complexity": "High",
            "category": "Advanced"
        },
        "hybrid": {
            "name": "Hybrid RAG",
            "description": "Combines multiple RAG approaches",
            "use_case": "Enterprise applications, complex domains",
            "complexity": "High",
            "category": "Enterprise"
        }
    }

@app.get("/models")
async def get_models():
    """Get available LLM models"""
    return {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "local": ["llama-3", "mistral", "phi-3"],
        "groq": ["llama-3-70b", "mixtral-8x7b"]
    }

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    try:
        info = rag_engine.get_session_info(session_id)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        rag_engine.delete_session(session_id)
        return {"status": "success", "message": "Session deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
