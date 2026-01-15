import os
import zipfile
from pathlib import Path
from typing import Dict

class CodeGenerator:
    def __init__(self):
        self.output_dir = Path("generated_code")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate(self, config, language: str = "python", include_frontend: bool = False) -> Dict[str, str]:
        """Generate complete RAG code"""
        
        if language == "python":
            code = self._generate_python_code(config, include_frontend)
        elif language == "javascript":
            code = self._generate_javascript_code(config, include_frontend)
        else:
            code = self._generate_python_code(config, include_frontend)
        
        return code
    
    def _generate_python_code(self, config, include_frontend: bool) -> Dict[str, str]:
        """Generate Python RAG implementation"""
        
        code = {}
        
        # Main application
        code['app.py'] = self._generate_python_app(config)
        
        # RAG implementation
        code['rag_system.py'] = self._generate_python_rag(config)
        
        # Requirements
        code['requirements.txt'] = self._generate_requirements(config)
        
        # Environment
        code['.env.example'] = self._generate_env_file(config)
        
        # README
        code['README.md'] = self._generate_readme(config)
        
        # Docker
        code['Dockerfile'] = self._generate_dockerfile()
        code['docker-compose.yml'] = self._generate_docker_compose(config)
        
        if include_frontend:
            code.update(self._generate_frontend())
        
        return code
    
    def _generate_python_app(self, config) -> str:
        return f'''from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from rag_system import RAGSystem

app = FastAPI(title="Generated RAG Application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGSystem(
    rag_type="{config.rag_type}",
    model_name="{config.model_name}",
    embedding_model="{config.embedding_model}"
)

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload and process documents"""
    documents = []
    for file in files:
        content = await file.read()
        documents.append(content.decode('utf-8'))
    
    rag.add_documents(documents)
    return {{"status": "success", "files_processed": len(files)}}

@app.post("/query")
async def query(query: str):
    """Query the RAG system"""
    response = rag.query(query)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _generate_python_rag(self, config) -> str:
        if config.rag_type == "vanilla":
            return self._generate_vanilla_rag_code(config)
        elif config.rag_type == "agentic":
            return self._generate_agentic_rag_code(config)
        elif config.rag_type == "graph":
            return self._generate_graph_rag_code(config)
        else:
            return self._generate_hybrid_rag_code(config)
    
    def _generate_vanilla_rag_code(self, config) -> str:
        return f'''from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

class RAGSystem:
    def __init__(self, rag_type, model_name, embedding_model):
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size={config.chunk_size},
            chunk_overlap={config.chunk_overlap}
        )
    
    def add_documents(self, documents):
        """Add documents to the RAG system"""
        texts = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc)
            texts.extend(chunks)
        
        self.vector_store = Chroma.from_texts(
            texts,
            self.embeddings,
            collection_name="rag_collection"
        )
        
        retriever = self.vector_store.as_retriever(search_kwargs={{"k": 5}})
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, query: str):
        """Query the RAG system"""
        if not self.qa_chain:
            return {{"error": "No documents loaded"}}
        
        result = self.qa_chain({{"query": query}})
        
        return {{
            "answer": result['result'],
            "sources": [doc.page_content for doc in result['source_documents']]
        }}
'''
    
    def _generate_agentic_rag_code(self, config) -> str:
        return f'''from langchain.agents import AgentExecutor, create_openai_functions_agent, Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

class RAGSystem:
    def __init__(self, rag_type, model_name, embedding_model):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.agent = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size={config.chunk_size},
            chunk_overlap={config.chunk_overlap}
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def add_documents(self, documents):
        """Add documents to the RAG system"""
        texts = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc)
            texts.extend(chunks)
        
        self.vector_store = Chroma.from_texts(
            texts,
            self.embeddings,
            collection_name="rag_collection"
        )
        
        retriever = self.vector_store.as_retriever()
        
        tools = [
            Tool(
                name="Knowledge Base",
                func=lambda q: retriever.get_relevant_documents(q),
                description="Search the knowledge base for information"
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to a knowledge base."),
            ("human", "{{input}}"),
            ("placeholder", "{{agent_scratchpad}}")
        ])
        
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, memory=self.memory, verbose=True)
    
    def query(self, query: str):
        """Query the RAG system"""
        if not self.agent:
            return {{"error": "No documents loaded"}}
        
        result = self.agent.invoke({{"input": query}})
        
        return {{
            "answer": result['output'],
            "reasoning": result.get('intermediate_steps', [])
        }}
'''
    
    def _generate_graph_rag_code(self, config) -> str:
        return f'''from langchain.graphs import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGSystem:
    def __init__(self, rag_type, model_name, embedding_model):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.graph = NetworkxEntityGraph()
        self.vector_store = None
        self.chain = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size={config.chunk_size},
            chunk_overlap={config.chunk_overlap}
        )
    
    def add_documents(self, documents):
        """Add documents to the RAG system"""
        texts = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc)
            texts.extend(chunks)
            
            # Extract entities and build graph
            entities = self._extract_entities(doc)
            for entity in entities:
                self.graph.add_triple(entity)
        
        self.vector_store = Chroma.from_texts(
            texts,
            self.embeddings,
            collection_name="rag_collection"
        )
        
        self.chain = GraphQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True
        )
    
    def _extract_entities(self, text):
        """Extract entities from text"""
        # Simplified entity extraction
        return []
    
    def query(self, query: str):
        """Query the RAG system"""
        if not self.chain:
            return {{"error": "No documents loaded"}}
        
        result = self.chain.run(query)
        
        return {{
            "answer": result,
            "graph_context": "Graph-based retrieval"
        }}
'''
    
    def _generate_hybrid_rag_code(self, config) -> str:
        return '''# Hybrid RAG combines multiple approaches
# Implementation combines vanilla + graph RAG
'''
    
    def _generate_requirements(self, config) -> str:
        base_reqs = '''fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
chromadb==0.4.18
openai==1.6.1
python-multipart==0.0.6
aiofiles==23.2.1
PyPDF2==3.0.1
python-docx==1.1.0
pandas==2.1.4
Pillow==10.1.0
pytesseract==0.3.10
moviepy==1.0.3
SpeechRecognition==3.10.0
pydub==0.25.1
sqlalchemy==2.0.23
'''
        
        if config.rag_type == "agentic":
            base_reqs += "langchain-experimental==0.0.47\n"
        
        if config.rag_type == "graph":
            base_reqs += "networkx==3.2.1\n"
        
        return base_reqs
    
    def _generate_env_file(self, config) -> str:
        return f'''# API Keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Model Configuration
MODEL_NAME={config.model_name}
EMBEDDING_MODEL={config.embedding_model}
RAG_TYPE={config.rag_type}

# Vector Database
VECTOR_DB={config.vector_db}
CHROMA_PERSIST_DIR=./chroma_db

# Application
APP_HOST=0.0.0.0
APP_PORT=8000
'''
    
    def _generate_readme(self, config) -> str:
        return f'''# Generated RAG Application

## Configuration
- RAG Type: {config.rag_type}
- Model: {config.model_name}
- Embedding: {config.embedding_model}

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and add your API keys.

## Usage

```bash
python app.py
```

API will be available at http://localhost:8000

## Endpoints

- POST /upload - Upload documents
- POST /query - Query the RAG system

## Docker

```bash
docker-compose up -d
```
'''
    
    def _generate_dockerfile(self) -> str:
        return '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
'''
    
    def _generate_docker_compose(self, config) -> str:
        return f'''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${{OPENAI_API_KEY}}
      - MODEL_NAME={config.model_name}
      - RAG_TYPE={config.rag_type}
    volumes:
      - ./uploads:/app/uploads
      - ./chroma_db:/app/chroma_db
'''
    
    def _generate_frontend(self) -> Dict[str, str]:
        """Generate React frontend"""
        return {
            'frontend/index.html': '''<!DOCTYPE html>
<html>
<head>
    <title>RAG System</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .query-box { width: 100%; padding: 10px; margin: 20px 0; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        .response { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel" src="app.js"></script>
</body>
</html>''',
            'frontend/app.js': '''const { useState } = React;

function App() {
    const [files, setFiles] = useState([]);
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState(null);
    
    const handleUpload = async () => {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        
        const res = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await res.json();
        alert('Files uploaded successfully!');
    };
    
    const handleQuery = async () => {
        const res = await fetch('http://localhost:8000/query?query=' + encodeURIComponent(query), {
            method: 'POST'
        });
        
        const data = await res.json();
        setResponse(data);
    };
    
    return (
        <div>
            <h1>RAG System</h1>
            
            <div className="upload-area">
                <input type="file" multiple onChange={(e) => setFiles(Array.from(e.target.files))} />
                <button onClick={handleUpload}>Upload Documents</button>
            </div>
            
            <input 
                className="query-box"
                type="text" 
                placeholder="Ask a question..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
            />
            <button onClick={handleQuery}>Query</button>
            
            {response && (
                <div className="response">
                    <h3>Answer:</h3>
                    <p>{response.answer}</p>
                </div>
            )}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));'''
        }
    
    def create_project_zip(self, session_id: str, code: Dict[str, str]) -> Path:
        """Create ZIP file with generated code"""
        project_dir = self.output_dir / session_id
        project_dir.mkdir(exist_ok=True)
        
        # Write all files
        for filename, content in code.items():
            file_path = project_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
        
        # Create ZIP
        zip_path = self.output_dir / f"{session_id}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(project_dir)
                    zipf.write(file_path, arcname)
        
        return zip_path
