import uuid
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma, FAISS, Milvus
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
import chromadb
from services.rag_implementations import RAGImplementations

class RAGEngine:
    def __init__(self):
        self.sessions = {}
        self.vector_stores = {}
        
    async def create_rag(self, session_id: str, config: Any, documents: List[Dict]):
        """Create RAG system based on type"""
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        texts = []
        for doc in documents:
            chunks = text_splitter.split_text(doc['content'])
            texts.extend(chunks)
        
        # Create embeddings
        embeddings = self._get_embeddings(config.embedding_model)
        
        # Create vector store
        vector_store = self._create_vector_store(
            config.vector_db,
            texts,
            embeddings,
            session_id
        )
        
        # Initialize RAG implementations
        rag_impl = RAGImplementations(config)
        
        # Create RAG based on type
        rag_type = config.rag_type
        
        if rag_type == "vanilla":
            rag = rag_impl.vanilla_rag(vector_store)
        elif rag_type == "parent_document":
            rag = rag_impl.parent_document_rag(texts)
        elif rag_type == "multi_query":
            rag = rag_impl.multi_query_rag(vector_store)
        elif rag_type == "contextual_compression":
            rag = rag_impl.contextual_compression_rag(vector_store)
        elif rag_type == "reranking":
            rag = rag_impl.reranking_rag(vector_store)
        elif rag_type == "ensemble":
            rag = rag_impl.ensemble_rag(vector_store, texts)
        elif rag_type == "agentic":
            rag = rag_impl.agentic_rag(vector_store)
        elif rag_type == "corrective":
            rag = rag_impl.corrective_rag(vector_store)
        elif rag_type == "self_rag":
            rag = rag_impl.self_rag(vector_store)
        elif rag_type == "adaptive":
            rag = rag_impl.adaptive_rag(vector_store)
        elif rag_type == "fusion":
            rag = rag_impl.fusion_rag(vector_store)
        elif rag_type == "hyde":
            rag = rag_impl.hyde_rag(vector_store)
        elif rag_type == "raptor":
            rag = rag_impl.raptor_rag(texts)
        elif rag_type == "graph":
            rag = rag_impl.graph_rag(documents)
        elif rag_type == "multimodal":
            rag = rag_impl.multimodal_rag(vector_store)
        elif rag_type == "temporal":
            rag = rag_impl.temporal_rag(vector_store)
        elif rag_type == "conversational":
            rag = rag_impl.conversational_rag(vector_store)
        elif rag_type == "streaming":
            rag = rag_impl.streaming_rag(vector_store)
        elif rag_type == "federated":
            rag = rag_impl.federated_rag([vector_store])
        elif rag_type == "hierarchical":
            rag = rag_impl.hierarchical_rag(texts)
        elif rag_type == "semantic_cache":
            rag = rag_impl.semantic_cache_rag(vector_store)
        elif rag_type == "active_rag":
            rag = rag_impl.active_rag(vector_store)
        elif rag_type == "hybrid":
            # Hybrid combines multiple approaches
            rag = {
                "vanilla": rag_impl.vanilla_rag(vector_store),
                "fusion": rag_impl.fusion_rag(vector_store),
                "reranking": rag_impl.reranking_rag(vector_store)
            }
        else:
            raise ValueError(f"Unknown RAG type: {rag_type}")
        
        self.sessions[session_id] = {
            "config": config,
            "rag": rag,
            "vector_store": vector_store,
            "documents": documents
        }
        
        return rag
    
    def _get_embeddings(self, model: str):
        """Get embedding model"""
        if model == "openai":
            return OpenAIEmbeddings()
        elif model == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings()
        else:
            return OpenAIEmbeddings()
    
    def _create_vector_store(self, db_type: str, texts: List[str], embeddings, session_id: str):
        """Create vector database"""
        if db_type == "chromadb":
            return Chroma.from_texts(
                texts,
                embeddings,
                collection_name=session_id
            )
        elif db_type == "faiss":
            return FAISS.from_texts(texts, embeddings)
        elif db_type == "milvus":
            return Milvus.from_texts(
                texts,
                embeddings,
                collection_name=session_id
            )
        else:
            return Chroma.from_texts(texts, embeddings)
    
    def _create_vanilla_rag(self, vector_store, config):
        """Create simple RAG"""
        llm = self._get_llm(config.model_name)
        
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    
    def _create_agentic_rag(self, vector_store, config):
        """Create agentic RAG with reasoning"""
        from langchain.agents import Tool
        from langchain.memory import ConversationBufferMemory
        
        llm = self._get_llm(config.model_name)
        
        retriever = vector_store.as_retriever()
        
        tools = [
            Tool(
                name="Knowledge Base",
                func=lambda q: retriever.get_relevant_documents(q),
                description="Search the knowledge base for information"
            )
        ]
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to a knowledge base. Use the tools to answer questions accurately."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
        
        return agent_executor
    
    def _create_graph_rag(self, vector_store, config, documents):
        """Create graph-based RAG"""
        from langchain.graphs import NetworkxEntityGraph
        from langchain.chains import GraphQAChain
        
        llm = self._get_llm(config.model_name)
        
        # Build knowledge graph
        graph = NetworkxEntityGraph()
        for doc in documents:
            # Extract entities and relationships
            entities = self._extract_entities(doc['content'], llm)
            for entity in entities:
                graph.add_triple(entity)
        
        # Create graph QA chain
        chain = GraphQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True
        )
        
        return {
            "chain": chain,
            "vector_store": vector_store,
            "graph": graph
        }
    
    def _create_hybrid_rag(self, vector_store, config, documents):
        """Create hybrid RAG combining multiple approaches"""
        vanilla = self._create_vanilla_rag(vector_store, config)
        graph = self._create_graph_rag(vector_store, config, documents)
        
        return {
            "vanilla": vanilla,
            "graph": graph,
            "vector_store": vector_store
        }
    
    def _get_llm(self, model_name: str):
        """Get LLM instance"""
        if "gpt" in model_name:
            return ChatOpenAI(model=model_name, temperature=0)
        elif "claude" in model_name:
            return ChatAnthropic(model=model_name, temperature=0)
        else:
            return ChatOpenAI(model="gpt-4", temperature=0)
    
    def _extract_entities(self, text: str, llm):
        """Extract entities and relationships from text"""
        prompt = f"""Extract entities and relationships from the following text.
        Return as list of (subject, predicate, object) triples.
        
        Text: {text}
        
        Triples:"""
        
        response = llm.predict(prompt)
        # Parse response into triples
        return []
    
    async def query(self, session_id: str, query: str, top_k: int = 5):
        """Query the RAG system"""
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session = self.sessions[session_id]
        rag = session['rag']
        config = session['config']
        
        if config.rag_type == "vanilla":
            result = rag({"query": query})
            return {
                "answer": result['result'],
                "sources": [doc.page_content for doc in result['source_documents']]
            }
        elif config.rag_type == "agentic":
            result = rag.invoke({"input": query})
            return {
                "answer": result['output'],
                "reasoning": result.get('intermediate_steps', [])
            }
        elif config.rag_type == "graph":
            result = rag['chain'].run(query)
            return {
                "answer": result,
                "graph_context": "Graph-based retrieval"
            }
        elif config.rag_type == "hybrid":
            vanilla_result = rag['vanilla']({"query": query})
            graph_result = rag['graph']['chain'].run(query)
            
            return {
                "answer": vanilla_result['result'],
                "graph_answer": graph_result,
                "sources": [doc.page_content for doc in vanilla_result['source_documents']]
            }
    
    def get_config(self, session_id: str):
        """Get session configuration"""
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        return self.sessions[session_id]['config']
    
    def get_session_info(self, session_id: str):
        """Get session information"""
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "rag_type": session['config'].rag_type,
            "model": session['config'].model_name,
            "documents_count": len(session['documents'])
        }
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.vector_stores:
            del self.vector_stores[session_id]
