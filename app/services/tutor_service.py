from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from typing import Dict, List
import json
from app.services.gemini_service import GeminiService

class TutorService:
    def __init__(self, vectorstore, model_name: str = "gemini"):
        self.vectorstore = vectorstore
        self.gemini_service = GeminiService()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Custom prompt for tutoring
        self.tutor_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
            You are an expert tutor for competitive exams like JEE, UPSC, GATE, etc.
            Use the following context to answer the student's question.
            Be helpful, encouraging, and provide step-by-step explanations when needed.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Student Question: {question}
            
            Answer as a helpful tutor:
            """
        )
    
    def ask_question(self, question: str, user_id: str = None) -> Dict:
        """Handle student questions with context retrieval"""
        try:
            # Get relevant context
            context_docs = self.vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in context_docs])
        except:
            context = "No relevant documents found in the knowledge base."
        
        # Generate response using Gemini
        prompt = f"""
        You are an educational tutor. Answer this question with examples and explanations:

            Context: {context}
            Question: {question}

            Include:
            - Clear explanation
            - Relevant examples
            - Key points to remember
            - Practice suggestions
            - And limit solution to 100 words only 

        """
        
        answer = self.gemini_service.generate_response(prompt, context)
        
        return {
            'answer': answer,
            'sources': [{'source': 'uploaded_documents', 'chunk_id': i} for i in range(len(context_docs))],
            'context_used': context[:500] + "..." if len(context) > 500 else context
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
    
    def provide_hint(self, question: str) -> str:
        """Provide hints without giving away the answer"""
        prompt = f"""
        Provide a helpful hint for the following question without giving away the complete answer:
        
        Question: {question}
        
        The hint should:
        1. Guide the student in the right direction
        2. Not reveal the complete answer
        3. Encourage critical thinking
        4. Be encouraging and supportive
        """
        
        return self.gemini_service.generate_response(prompt)
    
    def explain_concept(self, concept: str) -> str:
        """Explain a specific concept in detail"""
        prompt = f"""
        Explain the concept of "{concept}" in detail for competitive exam preparation:
        
        Include:
        1. Clear definition and explanation
        2. Key points and important aspects
        3. Real-world examples and applications
        4. Common misconceptions to avoid
        5. Practice tips and strategies
        6. Related concepts and connections
        
        Make it comprehensive yet easy to understand for students preparing for competitive exams.
        """
        
        return self.gemini_service.generate_response(prompt)