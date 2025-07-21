from typing import List, Dict
from app.services.gemini_service import GeminiService
import json

class MCPService:
    def __init__(self):
        self.gemini_service = GeminiService()
    
    def generate_mcqs(self, topic: str, context: str = "") -> List[Dict]:
        """Generate multiple choice questions using Gemini"""
        return self.gemini_service.generate_mcq(topic, context)
    
    def generate_summary(self, content: str) -> str:
        """Generate summary using Gemini"""
        return self.gemini_service.generate_summary(content)
    
    def generate_revision_notes(self, topic: str, content: str) -> str:
        """Generate mock revision notes"""
        return f"""
Revision Notes for {topic}:

1. Key Concepts:
   - Main topic: {topic}
   - Content length: {len(content)} characters
   - Word count: {len(content.split())} words

2. Important Points:
   - This is a mock revision note
   - The actual content would be analyzed for key concepts
   - Important formulas and definitions would be extracted

3. Common Mistakes to Avoid:
   - Not reading the full content
   - Missing key concepts
   - Poor time management

4. Practice Tips:
   - Review the content thoroughly
   - Create your own notes
   - Practice with similar questions
        """