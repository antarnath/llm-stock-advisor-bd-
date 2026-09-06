# PHASE 9 — RAG System

**Duration**: 2 Weeks  
**Started**: Week 18  
**Status**: 📝 Pending  
**Goal**: Build retrieval-augmented generation system

---

## 🎯 Objectives

1. Set up document processing pipeline
2. Implement vector storage with FAISS/ChromaDB
3. Build retrieval system
4. Integrate with LangChain
5. Create query interface

---

## 📚 Document Sources

- **Annual Reports**: PDF documents for 30+ companies
- **Quarterly Reports**: Quarterly financial statements
- **DSE Notices**: Official stock exchange notices
- **Company Disclosures**: Press releases, disclosures
- **Financial News**: News articles (processed in Phase 6)

---

## 🔧 Technical Stack

```
┌─────────────────────────────────────┐
│   Documents (PDFs, text)            │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Document Loader (PyPDF, etc.)     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Text Splitter (Chunking)          │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Embedding Model (OpenAI/HF)       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Vector Store (FAISS/ChromaDB)     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Retriever (Similarity Search)     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   LLM Generation (GPT-4/Claude)     │
└─────────────────────────────────────┘
```

---

## 📄 Component 1: Document Loader

```python
from langchain.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, documents_path='dataset/annual_reports/'):
        self.documents_path = documents_path
    
    def load_company_reports(self, company_code, year=None):
        """Load annual reports for a specific company"""
        if year:
            pdf_path = f"{self.documents_path}{company_code}/{year}.pdf"
            loader = PyPDFLoader(pdf_path)
        else:
            pdf_path = f"{self.documents_path}{company_code}/"
            loader = DirectoryLoader(
                pdf_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
        
        documents = loader.load()
        return documents
    
    def load_all_documents(self):
        """Load all documents in the dataset"""
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        return loader.load()
    
    def load_text_files(self):
        """Load text-based documents (announcements, etc.)"""
        from langchain.document_loaders import TextLoader
        
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        return loader.load()
```

---

## ✂️ Component 2: Text Chunking

```python
class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents):
        """Split documents into chunks"""
        chunks = self.splitter.split_documents(documents)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        return chunks
    
    def split_by_section(self, documents):
        """Split documents by sections (e.g., financial statements)"""
        section_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n\n", "\n\n", "\n"]
        )
        return section_splitter.split_documents(documents)
```

---

## 🧬 Component 3: Embeddings

```python
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings
)

class EmbeddingManager:
    def __init__(self, model_type='openai', model_name=None):
        if model_type == 'openai':
            self.embeddings = OpenAIEmbeddings(
                model=model_name or 'text-embedding-ada-002',
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        elif model_type == 'huggingface':
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name or 'sentence-transformers/all-MiniLM-L6-v2'
            )
        elif model_type == 'finbert':
            # Domain-specific embeddings for finance
            self.embeddings = HuggingFaceEmbeddings(
                model_name='ProsusAI/finbert'
            )
    
    def embed_documents(self, texts):
        """Generate embeddings for documents"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text):
        """Generate embedding for query"""
        return self.embeddings.embed_query(text)
```

---

## 💾 Component 4: Vector Store

### **FAISS Implementation**
```python
from langchain.vectorstores import FAISS
import faiss

class FAISSVectorStore:
    def __init__(self, embeddings, index_path='vectorstore/faiss_index'):
        self.embeddings = embeddings
        self.index_path = index_path
        self.vectorstore = None
    
    def create_index(self, documents):
        """Create FAISS index from documents"""
        self.vectorstore = FAISS.from_documents(
            documents, 
            self.embeddings
        )
        return self.vectorstore
    
    def save_index(self):
        """Save index to disk"""
        self.vectorstore.save_local(self.index_path)
    
    def load_index(self, allow_dangerous_deserialization=True):
        """Load index from disk"""
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        return self.vectorstore
    
    def add_documents(self, documents):
        """Add new documents to existing index"""
        self.vectorstore.add_documents(documents)
        self.save_index()
    
    def similarity_search(self, query, k=5):
        """Search for similar documents"""
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query, k=5):
        """Search with relevance scores"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def mmr_search(self, query, k=5, lambda_mult=0.5):
        """Maximal Marginal Relevance search"""
        results = self.vectorstore.max_marginal_relevance_search(
            query, 
            k=k, 
            lambda_mult=lambda_mult
        )
        return results
```

### **ChromaDB Implementation**
```python
from langchain.vectorstores import Chroma

class ChromaVectorStore:
    def __init__(self, embeddings, persist_directory='vectorstore/chroma'):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def create_index(self, documents, collection_name='financial_docs'):
        """Create ChromaDB index"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        self.vectorstore.persist()
        return self.vectorstore
    
    def load_index(self, collection_name='financial_docs'):
        """Load existing ChromaDB index"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        return self.vectorstore
    
    def search_by_metadata(self, query, metadata_filter, k=5):
        """Search with metadata filter"""
        results = self.vectorstore.similarity_search(
            query, 
            k=k,
            filter=metadata_filter
        )
        return results
```

---

## 🔍 Component 5: Advanced Retrieval

### **Hybrid Search (BM25 + Vector)**
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents, embeddings, alpha=0.5):
        # BM25 (keyword-based)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10
        
        # Vector (semantic)
        vectorstore = FAISS.from_documents(documents, embeddings)
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Ensemble
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[alpha, 1 - alpha]
        )
    
    def retrieve(self, query, k=5):
        """Hybrid retrieval"""
        results = self.ensemble_retriever.get_relevant_documents(query)
        return results[:k]
```

### **Query Rewriting**
```python
class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm
    
    def rewrite_query(self, original_query, context=None):
        """Rewrite query for better retrieval"""
        prompt = f"""Rewrite the following financial query to be more 
        specific and suitable for document retrieval.

        Original Query: {original_query}
        Context: {context or 'N/A'}
        
        Rewritten Query:"""
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def generate_sub_queries(self, complex_query):
        """Break down complex query into sub-queries"""
        prompt = f"""Break down this complex financial question into 3-5 
        simpler sub-questions that can be answered separately.

        Complex Query: {complex_query}
        
        Sub-questions:
        1."""
        
        response = self.llm.invoke(prompt)
        sub_queries = response.content.strip().split('\n')
        return sub_queries
```

### **Re-ranking**
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        """Re-rank documents using cross-encoder"""
        # Prepare pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Score
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1]
        
        # Return top-k
        reranked = [documents[i] for i in ranked_indices[:top_k]]
        return reranked
```

---

## 🤖 Component 6: Generation

### **QA Chain**
```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class RAGGenerator:
    def __init__(self, vectorstore, llm_model='gpt-4'):
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=0,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 5}
        )
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            input_variables=['context', 'question'],
            template="""You are a financial analyst expert on Bangladesh 
            stock market. Use the following context to answer the question. 
            If you don't know the answer, say so.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )
        
        # QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.prompt_template}
        )
    
    def query(self, question):
        """Answer a question using RAG"""
        result = self.qa_chain({'query': question})
        
        return {
            'answer': result['result'],
            'sources': [
                {
                    'content': doc.page_content[:200] + '...',
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'Unknown')
                }
                for doc in result['source_documents']
            ]
        }
    
    def query_with_confidence(self, question):
        """Query with confidence scoring"""
        result = self.query(question)
        
        # Calculate confidence based on source relevance
        sources = result['sources']
        if sources:
            avg_relevance = np.mean([
                doc.metadata.get('score', 0.5) 
                for doc in result.get('source_documents', [])
            ])
        else:
            avg_relevance = 0.0
        
        result['confidence'] = avg_relevance
        return result
```

### **Conversational RAG**
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ConversationalRAG:
    def __init__(self, vectorstore, llm_model='gpt-4'):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)
        
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def chat(self, question):
        """Multi-turn conversation"""
        result = self.qa_chain({'question': question})
        
        return {
            'answer': result['answer'],
            'chat_history': result['chat_history'],
            'sources': result['source_documents']
        }
    
    def clear_memory(self):
        """Reset conversation"""
        self.memory.clear()
```

---

## 📊 Document Processing Pipeline

```python
class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.loader = DocumentLoader(config['documents_path'])
        self.chunker = TextChunker(
            chunk_size=config.get('chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap', 200)
        )
        self.embeddings = EmbeddingManager(
            model_type=config.get('embedding_model', 'openai')
        )
        self.vectorstore = None
    
    def build_index(self, force_rebuild=False):
        """Build complete RAG index"""
        if not force_rebuild and self.index_exists():
            print("Loading existing index...")
            return self.load_index()
        
        print("Loading documents...")
        documents = self.loader.load_all_documents()
        print(f"Loaded {len(documents)} documents")
        
        print("Chunking documents...")
        chunks = self.chunker.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        print("Generating embeddings...")
        if self.config.get('vector_db') == 'faiss':
            self.vectorstore = FAISSVectorStore(self.embeddings)
        else:
            self.vectorstore = ChromaVectorStore(self.embeddings)
        
        self.vectorstore.create_index(chunks)
        self.vectorstore.save_index()
        
        print("✅ Index built successfully")
        return self.vectorstore
    
    def query(self, question, k=5):
        """Query the RAG system"""
        if not self.vectorstore:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # Generate answer
        generator = RAGGenerator(self.vectorstore)
        result = generator.query(question)
        
        return result
```

---

## 🗂️ Metadata Management

```python
class MetadataManager:
    """Manage document metadata for filtered retrieval"""
    
    @staticmethod
    def enrich_metadata(documents):
        """Add useful metadata to documents"""
        for doc in documents:
            source = doc.metadata.get('source', '')
            
            # Extract company code from filename
            if 'GP' in source.upper():
                doc.metadata['company'] = 'GP'
                doc.metadata['sector'] = 'Telecom'
            elif 'BATBC' in source.upper():
                doc.metadata['company'] = 'BATBC'
                doc.metadata['sector'] = 'Tobacco'
            # ... etc
            
            # Extract year from filename
            import re
            year_match = re.search(r'20\d{2}', source)
            if year_match:
                doc.metadata['year'] = int(year_match.group(0))
            
            # Document type
            if 'annual' in source.lower():
                doc.metadata['doc_type'] = 'annual_report'
            elif 'quarterly' in source.lower():
                doc.metadata['doc_type'] = 'quarterly_report'
            else:
                doc.metadata['doc_type'] = 'unknown'
        
        return documents
```

---

## 🧪 Evaluation

### **Retrieval Quality Metrics**
```python
class RetrievalEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, queries, ground_truth, retrieved_docs):
        """Evaluate retrieval quality"""
        # Precision@K
        precisions = []
        for query, truth, retrieved in zip(queries, ground_truth, retrieved_docs):
            relevant_docs = set(truth)
            retrieved_set = set([doc.metadata.get('id') for doc in retrieved])
            
            if len(retrieved_set) > 0:
                precision = len(relevant_docs & retrieved_set) / len(retrieved_set)
                precisions.append(precision)
        
        self.metrics['precision@5'] = np.mean(precisions)
        
        # Recall@K
        recalls = []
        for query, truth, retrieved in zip(queries, ground_truth, retrieved_docs):
            relevant_docs = set(truth)
            retrieved_set = set([doc.metadata.get('id') for doc in retrieved])
            
            if len(relevant_docs) > 0:
                recall = len(relevant_docs & retrieved_set) / len(relevant_docs)
                recalls.append(recall)
        
        self.metrics['recall@5'] = np.mean(recalls)
        
        # MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for query, truth, retrieved in zip(queries, ground_truth, retrieved_docs):
            relevant_docs = set(truth)
            for i, doc in enumerate(retrieved):
                if doc.metadata.get('id') in relevant_docs:
                    reciprocal_ranks.append(1 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0)
        
        self.metrics['MRR'] = np.mean(reciprocal_ranks)
        
        return self.metrics
```

### **Generation Quality**
```python
class GenerationEvaluator:
    def __init__(self):
        from datasets import load_metric
        self.bleu = load_metric('bleu')
        self.rouge = load_metric('rouge')
    
    def evaluate_answer_quality(self, generated_answers, reference_answers):
        """Evaluate answer generation quality"""
        results = {}
        
        # BLEU score
        bleu_scores = []
        for gen, ref in zip(generated_answers, reference_answers):
            score = self.bleu.compute(
                predictions=[gen.split()],
                references=[[ref.split()]]
            )
            bleu_scores.append(score['bleu'])
        results['BLEU'] = np.mean(bleu_scores)
        
        # ROUGE score
        rouge_scores = self.rouge.compute(
            predictions=generated_answers,
            references=reference_answers
        )
        results['ROUGE-L'] = rouge_scores['rougeL'].mid.fmeasure
        
        # Factuality (manual evaluation)
        results['factuality'] = self.manual_factuality_check(
            generated_answers, reference_answers
        )
        
        return results
```

---

## 📂 Project Structure

```
rag/
├── document_loader.py
├── text_chunker.py
├── embeddings.py
├── vector_store/
│   ├── faiss_store.py
│   └── chroma_store.py
├── retrievers/
│   ├── hybrid_retriever.py
│   ├── query_rewriter.py
│   └── reranker.py
├── generation/
│   ├── qa_chain.py
│   └── conversational_rag.py
├── pipeline.py
├── evaluator.py
├── metadata_manager.py
└── vector_db/
    ├── faiss_index/
    └── chroma_db/
```

---

## ✅ Success Criteria

- [ ] Document loading pipeline functional
- [ ] Text chunking optimized
- [ ] Embeddings generated (1000s of documents)
- [ ] Vector store created and persisted
- [ ] Retrieval system returns relevant docs
- [ ] QA chain generates accurate answers
- [ ] Conversational RAG working
- [ ] Retrieval quality > 80% precision@5
- [ ] Query interface accessible via API
- [ ] Documentation complete

---

## 🛠️ Tools & Libraries

- **LangChain**: RAG framework
- **FAISS**: Facebook AI Similarity Search
- **ChromaDB**: Vector database
- **OpenAI Embeddings**: text-embedding-ada-002
- **HuggingFace**: Alternative embeddings
- **PyPDF**: PDF parsing
- **Sentence-Transformers**: Re-ranking
- **BM25**: Keyword search

---

## 💡 Best Practices

1. **Chunk size 500-1500** tokens works best
2. **Overlap 10-20%** of chunk size
3. **Use metadata filters** for efficient retrieval
4. **Re-rank top results** for better precision
5. **Cache embeddings** to avoid recomputation
6. **Update index incrementally** for new documents
7. **Monitor retrieval quality** regularly

---

## 🔧 Example Usage

```python
# Build RAG system
config = {
    'documents_path': 'dataset/annual_reports/',
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'embedding_model': 'openai',
    'vector_db': 'faiss'
}

pipeline = RAGPipeline(config)
vectorstore = pipeline.build_index()

# Query
result = pipeline.query("What was GP's revenue in 2023?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

---

**Next Phase**: Phase 10 — Multi-Agent System

**Last Updated**: 2026-08-13
