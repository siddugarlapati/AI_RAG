// Learning Page JavaScript

let currentLesson = 1;

function showLesson(lessonNum) {
    // Hide all lessons
    document.querySelectorAll('.lesson').forEach(lesson => {
        lesson.classList.remove('active');
    });
    
    // Remove active from all buttons
    document.querySelectorAll('.lesson-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected lesson
    document.getElementById(`lesson-${lessonNum}`).classList.add('active');
    document.querySelectorAll('.lesson-btn')[lessonNum - 1].classList.add('active');
    
    // Update progress bar
    const progress = (lessonNum / 5) * 100;
    document.getElementById('progressBar').style.width = progress + '%';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    currentLesson = lessonNum;
}

function runDemo() {
    const output = document.getElementById('demoOutput');
    output.style.display = 'block';
    
    // Animate stages
    const stages = output.querySelectorAll('.demo-stage, .demo-result');
    stages.forEach((stage, index) => {
        stage.style.display = 'none';
        setTimeout(() => {
            stage.style.display = 'block';
        }, index * 800);
    });
}

function showRAGDetail(ragType) {
    const modal = document.getElementById('ragModal');
    const modalBody = document.getElementById('modalBody');
    
    const ragDetails = {
        vanilla: {
            title: '🍦 Vanilla RAG',
            description: 'The simplest and fastest RAG implementation',
            howItWorks: [
                'User asks a question',
                'Convert question to vector embedding',
                'Search vector database for similar chunks',
                'Retrieve top 5 most relevant chunks',
                'Send chunks + question to LLM',
                'LLM generates answer based on context'
            ],
            pros: [
                'Very fast response time',
                'Low cost',
                'Easy to implement',
                'Good for most use cases'
            ],
            cons: [
                'May miss nuanced information',
                'Single retrieval pass',
                'No self-correction'
            ],
            bestFor: [
                'Documentation Q&A',
                'FAQs',
                'Simple knowledge bases',
                'Getting started with RAG'
            ],
            example: 'Customer support chatbot answering product questions from documentation'
        },
        agentic: {
            title: '🤖 Agentic RAG',
            description: 'Multi-agent system with reasoning capabilities',
            howItWorks: [
                'User asks complex question',
                'Agent analyzes question complexity',
                'Agent decides retrieval strategy',
                'Multiple retrieval passes if needed',
                'Agent reasons over retrieved information',
                'Agent verifies answer quality',
                'Final answer with reasoning trace'
            ],
            pros: [
                'Handles complex queries',
                'Multi-step reasoning',
                'Self-verification',
                'Can use multiple tools'
            ],
            cons: [
                'Slower response time',
                'Higher cost',
                'More complex to set up'
            ],
            bestFor: [
                'Research assistants',
                'Complex analysis',
                'Multi-step problem solving',
                'Decision support systems'
            ],
            example: 'Financial analyst agent that retrieves market data, analyzes trends, and provides investment recommendations'
        },
        corrective: {
            title: '✅ Corrective RAG (CRAG)',
            description: 'Self-correcting RAG with web fallback',
            howItWorks: [
                'Retrieve documents from knowledge base',
                'Grade relevance of retrieved documents',
                'If relevant: use documents',
                'If not relevant: search the web',
                'Generate answer from best source',
                'Cite sources used'
            ],
            pros: [
                'Self-correcting',
                'Web fallback for missing info',
                'Higher accuracy',
                'Always finds answer'
            ],
            cons: [
                'More API calls',
                'Slower than vanilla',
                'Requires web search API'
            ],
            bestFor: [
                'Accuracy-critical applications',
                'Dynamic information needs',
                'Fact-checking systems',
                'News and current events'
            ],
            example: 'Medical information system that verifies answers against latest research'
        },
        fusion: {
            title: '🔀 RAG Fusion',
            description: 'Multiple query generation with reciprocal rank fusion',
            howItWorks: [
                'Generate 3-5 variations of user question',
                'Retrieve documents for each variation',
                'Apply reciprocal rank fusion algorithm',
                'Combine and rank all results',
                'Use top-ranked documents',
                'Generate comprehensive answer'
            ],
            pros: [
                'Very comprehensive results',
                'Multiple perspectives',
                'Better coverage',
                'Reduces missed information'
            ],
            cons: [
                'Multiple retrievals needed',
                'Higher cost',
                'Slower response'
            ],
            bestFor: [
                'Research applications',
                'Comprehensive search',
                'Multi-faceted questions',
                'Academic use cases'
            ],
            example: 'Research assistant that explores a topic from multiple angles'
        },
        graph: {
            title: '🕸️ Graph RAG',
            description: 'Knowledge graph-based retrieval',
            howItWorks: [
                'Build knowledge graph from documents',
                'Extract entities and relationships',
                'User asks question',
                'Identify relevant entities',
                'Traverse graph to find connections',
                'Generate answer with relationship context'
            ],
            pros: [
                'Understands relationships',
                'Great for connected data',
                'Finds indirect connections',
                'Rich context'
            ],
            cons: [
                'Complex graph building',
                'Requires entity extraction',
                'Higher setup cost'
            ],
            bestFor: [
                'Network analysis',
                'Relationship queries',
                'Social networks',
                'Knowledge bases with connections'
            ],
            example: 'Company org chart assistant that understands reporting relationships'
        },
        multimodal: {
            title: '🎭 Multimodal RAG',
            description: 'Handle text, images, audio, and video',
            howItWorks: [
                'Process all media types',
                'Extract text from images (OCR)',
                'Transcribe audio/video',
                'Create unified embeddings',
                'Search across all modalities',
                'Generate answer with media context'
            ],
            pros: [
                'Handles all content types',
                'Rich media search',
                'Comprehensive understanding',
                'Future-proof'
            ],
            cons: [
                'Complex processing',
                'Higher storage needs',
                'More expensive'
            ],
            bestFor: [
                'Video libraries',
                'Image + text search',
                'Multimedia content',
                'Educational platforms'
            ],
            example: 'Video tutorial platform that can answer questions about video content'
        },
        conversational: {
            title: '💬 Conversational RAG',
            description: 'Multi-turn conversations with memory',
            howItWorks: [
                'Maintain conversation history',
                'User asks follow-up question',
                'Combine with previous context',
                'Retrieve relevant information',
                'Generate contextual answer',
                'Update conversation memory'
            ],
            pros: [
                'Natural conversations',
                'Remembers context',
                'Follow-up questions work',
                'Better user experience'
            ],
            cons: [
                'Memory management needed',
                'Context window limits',
                'More complex state'
            ],
            bestFor: [
                'Chatbots',
                'Virtual assistants',
                'Customer support',
                'Interactive applications'
            ],
            example: 'Customer support chatbot that remembers previous questions in conversation'
        }
    };
    
    const detail = ragDetails[ragType];
    if (!detail) return;
    
    modalBody.innerHTML = `
        <h2>${detail.title}</h2>
        <p class="modal-description">${detail.description}</p>
        
        <div class="modal-section">
            <h3>📋 How It Works</h3>
            <ol class="modal-list">
                ${detail.howItWorks.map(step => `<li>${step}</li>`).join('')}
            </ol>
        </div>
        
        <div class="modal-section">
            <h3>✅ Pros</h3>
            <ul class="modal-list pros">
                ${detail.pros.map(pro => `<li>${pro}</li>`).join('')}
            </ul>
        </div>
        
        <div class="modal-section">
            <h3>❌ Cons</h3>
            <ul class="modal-list cons">
                ${detail.cons.map(con => `<li>${con}</li>`).join('')}
            </ul>
        </div>
        
        <div class="modal-section">
            <h3>🎯 Best For</h3>
            <ul class="modal-list">
                ${detail.bestFor.map(use => `<li>${use}</li>`).join('')}
            </ul>
        </div>
        
        <div class="modal-section example">
            <h3>💡 Example Use Case</h3>
            <p>${detail.example}</p>
        </div>
        
        <button class="modal-action-btn" onclick="window.location.href='index.html'">
            Build This RAG →
        </button>
    `;
    
    modal.style.display = 'block';
}

function closeModal() {
    document.getElementById('ragModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('ragModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}

function getRecommendation() {
    const usecase = document.querySelector('input[name="usecase"]:checked')?.value;
    const datatype = document.querySelector('input[name="datatype"]:checked')?.value;
    const priority = document.querySelector('input[name="priority"]:checked')?.value;
    
    if (!usecase || !datatype || !priority) {
        alert('Please answer all questions!');
        return;
    }
    
    let recommendation = '';
    let ragType = '';
    let icon = '';
    
    // Decision logic
    if (usecase === 'simple' && priority === 'speed') {
        ragType = 'Vanilla RAG';
        icon = '🍦';
        recommendation = 'Perfect for your needs! Vanilla RAG is fast, simple, and cost-effective for basic Q&A.';
    } else if (usecase === 'research' || priority === 'comprehensive') {
        ragType = 'RAG Fusion';
        icon = '🔀';
        recommendation = 'RAG Fusion will give you comprehensive results from multiple angles, perfect for research!';
    } else if (usecase === 'chatbot') {
        ragType = 'Conversational RAG';
        icon = '💬';
        recommendation = 'Conversational RAG maintains context across multiple turns, ideal for chatbots!';
    } else if (usecase === 'enterprise' || priority === 'accuracy') {
        ragType = 'Corrective RAG (CRAG)';
        icon = '✅';
        recommendation = 'Corrective RAG ensures maximum accuracy with self-correction and web fallback!';
    } else if (datatype === 'database') {
        ragType = 'SQL RAG';
        icon = '🗄️';
        recommendation = 'SQL RAG converts natural language to SQL queries, perfect for databases!';
    } else if (datatype === 'media') {
        ragType = 'Multimodal RAG';
        icon = '🎭';
        recommendation = 'Multimodal RAG handles images, audio, and video alongside text!';
    } else if (priority === 'reasoning') {
        ragType = 'Agentic RAG';
        icon = '🤖';
        recommendation = 'Agentic RAG provides multi-step reasoning for complex queries!';
    } else {
        ragType = 'Reranking RAG';
        icon = '🎯';
        recommendation = 'Reranking RAG balances speed and accuracy with two-stage retrieval!';
    }
    
    const recBox = document.getElementById('recommendation');
    recBox.innerHTML = `
        <div class="rec-icon">${icon}</div>
        <h3>We Recommend: ${ragType}</h3>
        <p class="rec-description">${recommendation}</p>
        <div class="rec-actions">
            <button class="rec-btn primary" onclick="window.location.href='index.html'">
                Build ${ragType} Now →
            </button>
            <button class="rec-btn secondary" onclick="showRAGDetail('${ragType.toLowerCase().replace(/ /g, '_').replace(/\(|\)/g, '')}')">
                Learn More About ${ragType}
            </button>
        </div>
    `;
    recBox.style.display = 'block';
    
    // Scroll to recommendation
    recBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Add styles for modal content
const style = document.createElement('style');
style.textContent = `
    .modal-description {
        font-size: 1.2em;
        color: #666;
        margin: 20px 0;
    }
    
    .modal-section {
        margin: 30px 0;
        padding: 20px;
        background: #f9f9f9;
        border-radius: 10px;
    }
    
    .modal-section h3 {
        margin-bottom: 15px;
        color: #333;
    }
    
    .modal-list {
        padding-left: 20px;
    }
    
    .modal-list li {
        margin: 10px 0;
        line-height: 1.6;
    }
    
    .modal-list.pros li::marker {
        color: #4caf50;
    }
    
    .modal-list.cons li::marker {
        color: #f44336;
    }
    
    .modal-section.example {
        background: linear-gradient(135deg, #e8f0fe, #c8e6ff);
        border-left: 4px solid #667eea;
    }
    
    .modal-action-btn {
        width: 100%;
        padding: 15px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        margin-top: 20px;
    }
    
    .rec-icon {
        font-size: 4em;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .rec-description {
        font-size: 1.2em;
        line-height: 1.6;
        margin: 20px 0;
    }
    
    .rec-actions {
        display: flex;
        gap: 15px;
        margin-top: 30px;
    }
    
    .rec-btn {
        flex: 1;
        padding: 15px;
        border: none;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
    }
    
    .rec-btn.primary {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    .rec-btn.secondary {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
    }
`;
document.head.appendChild(style);
