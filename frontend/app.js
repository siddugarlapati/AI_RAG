const API_URL = 'http://localhost:8000';
let selectedFiles = [];
let currentSessionId = null;

// RAG type descriptions
const ragDescriptions = {
    // Basic
    vanilla: 'Simple and fast retrieval-augmented generation. Best for basic Q&A and documentation search.',
    parent_document: 'Retrieve small chunks but return full parent documents for better context.',
    
    // Intermediate
    multi_query: 'Generate multiple query variations for comprehensive retrieval from different angles.',
    contextual_compression: 'Compress retrieved context to most relevant parts, reducing tokens and costs.',
    reranking: 'Two-stage retrieval with reranking for improved relevance and precision.',
    ensemble: 'Combine multiple retrievers (vector + BM25) for better coverage.',
    
    // Advanced
    agentic: 'Multi-agent system with reasoning capabilities. Handles complex queries with multi-step thinking.',
    corrective: 'Self-correcting RAG with web fallback. Grades relevance and uses web search if needed.',
    self_rag: 'Self-reflective RAG that assesses and improves its own answers.',
    adaptive: 'Dynamically adjusts retrieval strategy based on query complexity.',
    fusion: 'Generate multiple queries and use reciprocal rank fusion for better results.',
    hyde: 'Generate hypothetical answers first, then use them for better semantic retrieval.',
    raptor: 'Recursive abstractive processing with tree organization for long documents.',
    
    // Specialized
    graph: 'Knowledge graph-based retrieval. Perfect for relationship queries and connected data.',
    sql: 'Natural language to SQL queries for database interaction.',
    multimodal: 'Handle text, images, audio, and video content in a unified system.',
    temporal: 'Time-aware retrieval for historical data and time-series queries.',
    conversational: 'Multi-turn conversations with memory for chatbot applications.',
    streaming: 'Real-time streaming responses for live updates.',
    
    // Enterprise
    federated: 'Query across multiple distributed data sources simultaneously.',
    hierarchical: 'Multi-level retrieval strategy for structured documents and taxonomies.',
    hybrid: 'Combines multiple RAG approaches. Enterprise-grade solution for complex domains.',
    
    // Optimization
    semantic_cache: 'Cache similar queries semantically for improved performance.',
    active_rag: 'Active learning with user feedback for continuous improvement.'
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateRagDescription();
});

function setupEventListeners() {
    // File upload
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.background = '#f8f9ff';
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.background = '';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.background = '';
        handleFiles(e.dataTransfer.files);
    });
    
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    // RAG type change
    document.getElementById('ragType').addEventListener('change', updateRagDescription);
    
    // Chunk size slider
    document.getElementById('chunkSize').addEventListener('input', (e) => {
        document.getElementById('chunkSizeValue').textContent = e.target.value;
    });
}

function handleFiles(files) {
    selectedFiles = Array.from(files);
    displayFiles();
}

function displayFiles() {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>${file.name} (${formatFileSize(file.size)})</span>
            <button onclick="removeFile(${index})">Remove</button>
        `;
        fileList.appendChild(fileItem);
    });
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    displayFiles();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function updateRagDescription() {
    const ragType = document.getElementById('ragType').value;
    const description = document.getElementById('ragDescription');
    description.textContent = ragDescriptions[ragType];
}

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

async function uploadFiles() {
    if (selectedFiles.length === 0) {
        showNotification('Please select files to upload', 'error');
        return;
    }
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    formData.append('rag_type', document.getElementById('ragType').value);
    formData.append('model_name', document.getElementById('modelName').value);
    formData.append('chunk_size', document.getElementById('chunkSize').value);
    formData.append('embedding_model', document.getElementById('embeddingModel').value);
    
    try {
        showNotification('Creating RAG system...', 'info');
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentSessionId = data.session_id;
            document.getElementById('sessionId').value = currentSessionId;
            document.getElementById('codeSessionId').value = currentSessionId;
            
            showNotification(`RAG system created! Session ID: ${currentSessionId}`, 'success');
            
            // Switch to query tab
            setTimeout(() => {
                document.querySelector('[onclick="showTab(\'query\')"]').click();
            }, 2000);
        } else {
            showNotification('Error: ' + data.detail, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

async function connectDatabase() {
    const dbType = document.getElementById('dbType').value;
    const connectionString = document.getElementById('connectionString').value;
    const tables = document.getElementById('tables').value;
    
    if (!connectionString) {
        showNotification('Please enter connection string', 'error');
        return;
    }
    
    try {
        showNotification('Connecting to database...', 'info');
        
        const response = await fetch(`${API_URL}/connect-database`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                db_type: dbType,
                connection_string: connectionString,
                tables: tables ? tables.split(',').map(t => t.trim()) : null
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentSessionId = data.session_id;
            document.getElementById('sessionId').value = currentSessionId;
            document.getElementById('codeSessionId').value = currentSessionId;
            
            showNotification(`Database RAG created! Session ID: ${currentSessionId}`, 'success');
            
            setTimeout(() => {
                document.querySelector('[onclick="showTab(\'query\')"]').click();
            }, 2000);
        } else {
            showNotification('Error: ' + data.detail, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

async function queryRAG() {
    const sessionId = document.getElementById('sessionId').value;
    const query = document.getElementById('queryInput').value;
    
    if (!sessionId || !query) {
        showNotification('Please enter session ID and query', 'error');
        return;
    }
    
    try {
        showNotification('Querying RAG system...', 'info');
        
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                query: query,
                top_k: 5
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResponse(data);
            showNotification('Query completed!', 'success');
        } else {
            showNotification('Error: ' + data.detail, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

function displayResponse(data) {
    const responseBox = document.getElementById('response');
    responseBox.classList.add('show');
    
    let html = '<h3>Answer:</h3>';
    html += `<div class="answer">${data.answer}</div>`;
    
    if (data.sources && data.sources.length > 0) {
        html += '<div class="sources"><h4>Sources:</h4>';
        data.sources.forEach((source, index) => {
            html += `<div class="source-item"><strong>Source ${index + 1}:</strong> ${source}</div>`;
        });
        html += '</div>';
    }
    
    if (data.reasoning) {
        html += '<div class="sources"><h4>Reasoning Steps:</h4>';
        html += `<div class="source-item">${JSON.stringify(data.reasoning, null, 2)}</div>`;
        html += '</div>';
    }
    
    responseBox.innerHTML = html;
}

async function generateCode() {
    const sessionId = document.getElementById('codeSessionId').value;
    const language = document.getElementById('language').value;
    const includeFrontend = document.getElementById('includeFrontend').checked;
    
    if (!sessionId) {
        showNotification('Please enter session ID', 'error');
        return;
    }
    
    try {
        showNotification('Generating code...', 'info');
        
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('language', language);
        formData.append('include_frontend', includeFrontend);
        
        const response = await fetch(`${API_URL}/generate-code`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `rag_project_${sessionId}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showNotification('Code generated and downloaded!', 'success');
        } else {
            const data = await response.json();
            showNotification('Error: ' + data.detail, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

function showNotification(message, type) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification show ${type}`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 5000);
}
