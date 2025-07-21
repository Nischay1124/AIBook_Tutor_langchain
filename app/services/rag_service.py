from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import chromadb
import os





class RAGService:
    def __init__(self, persist_directory: str = "data/embeddings"):
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use HuggingFace embeddings for vector storage
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}  # Use CPU to avoid GPU issues
            )
        except Exception as e:
            print(f"Error loading embeddings model: {e}")
            # Fallback to a simpler model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
            )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        self.initialize_vectorstore()
    
    def initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Vector store initialized at: {self.persist_directory}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # Try to create a new one
            try:
                import shutil
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
                
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print("Vector store recreated successfully")
            except Exception as e2:
                print(f"Failed to recreate vector store: {e2}")
                raise
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store"""
        try:
            # Split documents into chunks
            texts = []
            metadatas = []
            
            for doc in documents:
                if not doc.get('content'):
                    print(f"Warning: Document {doc.get('file_name', 'unknown')} has no content")
                    continue
                    
                chunks = self.text_splitter.split_text(doc['content'])
                for i, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metadatas.append({
                        'source': doc['file_path'],
                        'chunk_id': i,
                        'file_type': doc['file_type']
                    })
            
            if texts:
                # Add to vector store
                self.vectorstore.add_texts(texts, metadatas)
                self.vectorstore.persist()
                print(f"Added {len(texts)} text chunks to vector store")
            else:
                print("No text chunks to add")
                
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def get_context_for_question(self, question: str) -> str:
        """Get relevant context for a specific question"""
        try:
            results = self.search(question, k=3)
            context = "\n\n".join([r['content'] for r in results])
            return context
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""


