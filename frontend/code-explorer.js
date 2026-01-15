// Code Explorer JavaScript

const ragCodeExamples = {
    vanilla: {
        icon: '🍦',
        title: 'Vanilla RAG',
        description: 'Simple and fast retrieval-augmented generation. Perfect for getting started with RAG.',
        architecture: `
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Convert to │
│  Embedding  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Search    │
│  Vector DB  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieve   │
│  Top 5 Docs │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     LLM     │
│  Generate   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
        `,
        features: [
            'Fast response time (< 2s)',
            'Low cost per query',
            'Easy to implement',
            'Works with any LLM',
            'Scalable to millions of documents',
            'Simple to maintain'
        ],
        useCases: [
            {
                title: 'Documentation Q&A',
                description: 'Answer questions from product documentation, user manuals, or technical guides.'
            },
            {
                title: 'FAQ Chatbot',
                description: 'Automated customer support answering frequently asked questions.'
            },
            {
                title: 'Knowledge Base Search',
                description: 'Search through company knowledge bases and internal wikis.'
            }
        ],
        code: {
            'app.py': `from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

# Initialize components
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0)
vector_store = None

@app.post("/upload")
async def upload_documents(documents: list[str]):
    """Upload and index documents"""
    global vector_store
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        texts.extend(chunks)
    
    # Create vector store
    vector_store = Chroma.from_texts(
        texts,
        embeddings,
        collection_name="documents"
    )
    
    return {"status": "success", "chunks": len(texts)}

@app.post("/query")
async def query(question: str):
    """Query the RAG system"""
    if not vector_store:
        return {"error": "No documents uploaded"}
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Get answer
    result = qa_chain({"query": question})
    
    return {
        "answer": result['result'],
        "sources": [doc.page_content for doc in result['source_documents']]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`,
            'requirements.txt': `fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
chromadb==0.4.18
openai==1.6.1`,
            'README.md': `# Vanilla RAG Implementation

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Configuration

Set your OpenAI API key:
\`\`\`bash
export OPENAI_API_KEY=your-key-here
\`\`\`

## Run

\`\`\`bash
python app.py
\`\`\`

## Usage

Upload documents:
\`\`\`bash
curl -X POST http://localhost:8000/upload \\
  -H "Content-Type: application/json" \\
  -d '{"documents": ["Your document text here"]}'
\`\`\`

Query:
\`\`\`bash
curl -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What is...?"}'
\`\`\``
        }
    },
    
    agentic: {
        icon: '🤖',
        title: 'Agentic RAG',
        description: 'Multi-agent system with reasoning capabilities. Handles complex queries with multi-step thinking.',
        architecture: `
┌─────────────┐
│   Complex   │
│   Question  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Agent     │
│  Analyzes   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Plans     │
│  Strategy   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Multiple   │
│ Retrievals  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reasoning  │
│    Over     │
│   Results   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Verifies   │
│   Answer    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Final Answer │
└─────────────┘
        `,
        features: [
            'Multi-step reasoning',
            'Self-verification',
            'Tool usage (search, calculator, etc.)',
            'Handles complex queries',
            'Reasoning trace available',
            'Adaptive strategy'
        ],
        useCases: [
            {
                title: 'Research Assistant',
                description: 'Analyze multiple sources, synthesize information, and provide comprehensive answers.'
            },
            {
                title: 'Financial Analysis',
                description: 'Retrieve market data, analyze trends, and provide investment recommendations.'
            },
            {
                title: 'Legal Research',
                description: 'Search case law, analyze precedents, and provide legal opinions.'
            }
        ],
        code: {
            'app.py': `from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_openai_functions_agent, Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# Initialize
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0)
vector_store = None
agent_executor = None

@app.post("/upload")
async def upload_documents(documents: list[str]):
    """Upload and create agent"""
    global vector_store, agent_executor
    
    # Create vector store
    vector_store = Chroma.from_texts(
        documents,
        embeddings,
        collection_name="documents"
    )
    
    # Create retriever tool
    retriever = vector_store.as_retriever()
    
    tools = [
        Tool(
            name="Knowledge Base",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Search the knowledge base for information"
        )
    ]
    
    # Create agent
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with access to a knowledge base. "
                   "Use the tools to answer questions accurately. "
                   "Think step by step and verify your answers."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    
    return {"status": "success"}

@app.post("/query")
async def query(question: str):
    """Query the agentic RAG system"""
    if not agent_executor:
        return {"error": "No documents uploaded"}
    
    result = agent_executor.invoke({"input": question})
    
    return {
        "answer": result['output'],
        "reasoning": result.get('intermediate_steps', [])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`,
            'requirements.txt': `fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
chromadb==0.4.18
openai==1.6.1`,
            'README.md': `# Agentic RAG Implementation

Multi-agent system with reasoning capabilities.

## Features
- Multi-step reasoning
- Self-verification
- Tool usage
- Memory across conversations

## Setup & Run
Same as Vanilla RAG, but with agent capabilities!`
        }
    },
    
    // Add more RAG types here...
    corrective: {
        icon: '✅',
        title: 'Corrective RAG (CRAG)',
        description: 'Self-correcting RAG with web fallback. Ensures accuracy by grading relevance and using web search when needed.',
        architecture: `
┌─────────────┐
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieve   │
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Grade    │
│  Relevance  │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
Relevant  Not Relevant
   │       │
   │       ▼
   │  ┌─────────────┐
   │  │ Web Search  │
   │  └──────┬──────┘
   │         │
   └────┬────┘
        │
        ▼
   ┌─────────────┐
   │  Generate   │
   │   Answer    │
   └─────────────┘
        `,
        features: [
            'Self-correcting mechanism',
            'Web search fallback',
            'Relevance grading',
            'Higher accuracy',
            'Source citation',
            'Confidence scores'
        ],
        useCases: [
            {
                title: 'Medical Information',
                description: 'Verify medical information against latest research and guidelines.'
            },
            {
                title: 'News & Current Events',
                description: 'Combine internal knowledge with latest web information.'
            },
            {
                title: 'Fact-Checking',
                description: 'Verify claims against multiple sources including web.'
            }
        ],
        code: {
            'app.py': `from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import DuckDuckGoSearchRun

app = FastAPI()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0)
search = DuckDuckGoSearchRun()
vector_store = None

class CorrectiveRAG:
    def __init__(self, retriever, llm, search):
        self.retriever = retriever
        self.llm = llm
        self.search = search
    
    def __call__(self, query):
        # Retrieve documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Grade relevance
        grader_prompt = f"""Are these documents relevant to '{query}'?
        Answer yes or no.
        
        Documents: {docs}"""
        
        grade = self.llm.predict(grader_prompt).lower()
        
        if "yes" in grade:
            # Use retrieved docs
            context = "\\n".join([d.page_content for d in docs])
            used_web = False
        else:
            # Fallback to web search
            context = self.search.run(query)
            used_web = True
        
        # Generate answer
        answer_prompt = f"""Context: {context}
        
        Question: {query}
        
        Answer:"""
        
        answer = self.llm.predict(answer_prompt)
        
        return {
            "answer": answer,
            "sources": docs if not used_web else ["Web Search"],
            "used_web": used_web
        }

@app.post("/upload")
async def upload_documents(documents: list[str]):
    global vector_store
    vector_store = Chroma.from_texts(documents, embeddings)
    return {"status": "success"}

@app.post("/query")
async def query(question: str):
    if not vector_store:
        return {"error": "No documents uploaded"}
    
    rag = CorrectiveRAG(
        vector_store.as_retriever(),
        llm,
        search
    )
    
    result = rag(question)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`,
            'requirements.txt': `fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
chromadb==0.4.18
openai==1.6.1
duckduckgo-search==4.1.0`,
            'README.md': `# Corrective RAG (CRAG)

Self-correcting RAG with web fallback.

## Features
- Grades document relevance
- Falls back to web search
- Higher accuracy
- Source tracking

## Setup
Requires DuckDuckGo search tool.
No API key needed for search!`
        }
    }
};

function showCode(ragType) {
    const codeView = document.getElementById('code-view');
    const ragData = ragCodeExamples[ragType];
    
    if (!ragData) {
        codeView.innerHTML = '<p>Code example coming soon!</p>';
        return;
    }
    
    // Update active state
    document.querySelectorAll('.rag-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Build code view
    let html = `
        <div class="code-header">
            <div class="code-title">
                <div class="code-title-icon">${ragData.icon}</div>
                <h2>${ragData.title}</h2>
            </div>
            <p class="code-description">${ragData.description}</p>
            
            <div class="code-actions">
                <button class="action-btn primary" onclick="generateCustomCode('${ragType}')">
                    ⚙️ Generate Custom Code
                </button>
                <button class="action-btn secondary" onclick="downloadCode('${ragType}')">
                    📥 Download Project
                </button>
                <button class="action-btn secondary" onclick="window.location.href='index.html'">
                    🚀 Build Now
                </button>
            </div>
        </div>
        
        <div class="architecture-section">
            <h3>🏗️ Architecture</h3>
            <div class="architecture-diagram">
                <pre>${ragData.architecture}</pre>
            </div>
        </div>
        
        <div class="features-section">
            <h3>✨ Key Features</h3>
            <div class="features-grid">
                ${ragData.features.map(feature => `
                    <div class="feature-item">
                        <span class="feature-check">✓</span>
                        <span>${feature}</span>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="usecases-section">
            <h3>💡 Use Cases</h3>
            ${ragData.useCases.map(usecase => `
                <div class="usecase-item">
                    <h4>${usecase.title}</h4>
                    <p>${usecase.description}</p>
                </div>
            `).join('')}
        </div>
        
        <div class="code-section">
            <h3>💻 Complete Code</h3>
            <div class="code-tabs">
                ${Object.keys(ragData.code).map((filename, index) => `
                    <button class="code-tab ${index === 0 ? 'active' : ''}" 
                            onclick="showCodeTab('${ragType}', '${filename}')">
                        ${filename}
                    </button>
                `).join('')}
            </div>
            
            ${Object.entries(ragData.code).map(([filename, code], index) => `
                <div class="code-content ${index === 0 ? 'active' : ''}" id="${ragType}-${filename}">
                    <div class="code-block">
                        <div class="code-header-bar">
                            <span class="code-filename">${filename}</span>
                            <button class="copy-btn" onclick="copyCode('${ragType}', '${filename}')">
                                Copy
                            </button>
                        </div>
                        <pre><code>${escapeHtml(code)}</code></pre>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    codeView.innerHTML = html;
    
    // Scroll to top
    codeView.scrollTop = 0;
}

function showCodeTab(ragType, filename) {
    // Remove active from all tabs
    document.querySelectorAll('.code-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active from all content
    document.querySelectorAll('.code-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Activate selected
    event.target.classList.add('active');
    document.getElementById(`${ragType}-${filename}`).classList.add('active');
}

function copyCode(ragType, filename) {
    const ragData = ragCodeExamples[ragType];
    const code = ragData.code[filename];
    
    navigator.clipboard.writeText(code).then(() => {
        event.target.textContent = 'Copied!';
        setTimeout(() => {
            event.target.textContent = 'Copy';
        }, 2000);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function generateCustomCode(ragType) {
    const codeView = document.getElementById('code-view');
    codeView.innerHTML += `
        <div class="generate-form">
            <h3>⚙️ Generate Custom Code</h3>
            <p>Customize your ${ragType} implementation</p>
            
            <div class="form-group">
                <label>Project Name:</label>
                <input type="text" id="projectName" placeholder="my-rag-project">
            </div>
            
            <div class="form-group">
                <label>LLM Model:</label>
                <select id="llmModel">
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    <option value="claude-3-opus">Claude 3 Opus</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Vector Database:</label>
                <select id="vectorDb">
                    <option value="chromadb">ChromaDB</option>
                    <option value="faiss">FAISS</option>
                    <option value="milvus">Milvus</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" id="includeFrontend">
                    Include React Frontend
                </label>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" id="includeDocker">
                    Include Docker Configuration
                </label>
            </div>
            
            <button class="generate-btn" onclick="downloadCustomCode('${ragType}')">
                Generate & Download
            </button>
        </div>
    `;
    
    // Scroll to form
    document.querySelector('.generate-form').scrollIntoView({ behavior: 'smooth' });
}

function downloadCustomCode(ragType) {
    const projectName = document.getElementById('projectName').value || 'my-rag-project';
    const llmModel = document.getElementById('llmModel').value;
    const vectorDb = document.getElementById('vectorDb').value;
    const includeFrontend = document.getElementById('includeFrontend').checked;
    const includeDocker = document.getElementById('includeDocker').checked;
    
    // In real implementation, this would call the backend API
    alert(`Generating custom ${ragType} code with:
- Project: ${projectName}
- Model: ${llmModel}
- Vector DB: ${vectorDb}
- Frontend: ${includeFrontend}
- Docker: ${includeDocker}

This will be downloaded as a ZIP file!`);
    
    // Redirect to main platform for actual generation
    window.location.href = `index.html?generate=${ragType}`;
}

function downloadCode(ragType) {
    alert(`Downloading ${ragType} project...
    
This will download a complete project with:
- Source code
- Requirements
- README
- Docker configuration
- Example data

Redirecting to main platform...`);
    
    window.location.href = 'index.html';
}
