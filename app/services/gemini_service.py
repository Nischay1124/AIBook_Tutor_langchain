import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, List, Optional
import json
from app.config import Config

class GeminiService:
    def __init__(self):
        # Check if API key is configured
        if not Config.GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not found in environment variables")
            self.use_gemini = False
            return
        
        try:
            # Configure Gemini API
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            self.chat_model = ChatGoogleGenerativeAI(
                model=Config.GEMINI_MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.7,
                max_output_tokens=2048
            )
            self.use_gemini = True
            print(f"Gemini service initialized successfully with model: {Config.GEMINI_MODEL}")
        except Exception as e:
            print(f"Error initializing Gemini service: {e}")
            self.use_gemini = False
        
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate a response using Gemini"""
        if not self.use_gemini:
            return f"Mock Gemini Response: {prompt[:100]}... (API key not configured or service not initialized)"
        
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat_response(self, messages: List[Dict]) -> str:
        """Generate chat response using LangChain integration"""
        if not self.use_gemini:
            return "Mock chat response (API key not configured)"
        
        try:
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
            
            response = self.chat_model.invoke(langchain_messages)
            return response.content
        except Exception as e:
            print(f"Error in chat response: {e}")
            return f"Error in chat response: {str(e)}"
    
    def generate_mcq(self, topic: str, context: str = "") -> List[Dict]:
        """Generate multiple choice questions"""
        if not self.use_gemini:
            # Return mock MCQs if Gemini is not available
            return [
                {
                    "question": f"What is the main concept of {topic}?",
                    "options": [
                        "A concept in technology",
                        "A mathematical formula", 
                        "A programming language",
                        "A hardware component"
                    ],
                    "correct_answer": "A",
                    "explanation": "This is a mock explanation since Gemini API is not configured."
                },
                {
                    "question": f"Which of the following is related to {topic}?",
                    "options": [
                        "Cooking recipes",
                        "Data analysis",
                        "Sports equipment",
                        "Musical instruments"
                    ],
                    "correct_answer": "B",
                    "explanation": "This is a mock explanation since Gemini API is not configured."
                }
            ]
        
        # Shorter, more concise prompt
        prompt = f"""
        Generate 5 MCQs for topic: {topic}
        Context: {context}
        
        Return ONLY this JSON format:
        [
            {{
                "question": "Question text here",
                "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
                "correct_answer": "A",
                "explanation": "Brief explanation"
            }}
        ]
        
        Rules: A=first option, B=second, etc. Make options complete sentences.
        """
        
        try:
            response = self.generate_response(prompt)
            print(f"Raw Gemini Response: {response}")
            
            # Clean the response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Try to parse JSON response
            mcqs = json.loads(cleaned_response)
            
            # Validate the structure
            if not isinstance(mcqs, list):
                raise ValueError("Response is not a list")
            
            for mcq in mcqs:
                required_fields = ["question", "options", "correct_answer", "explanation"]
                for field in required_fields:
                    if field not in mcq:
                        raise ValueError(f"Missing field: {field}")
                
                # Ensure options is an array with actual text
                if not isinstance(mcq["options"], list) or len(mcq["options"]) != 4:
                    raise ValueError("Options must be an array with exactly 4 elements")
            
            return mcqs
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Cleaned response: {cleaned_response}")
            
            # Try to extract JSON from the response
            try:
                # Look for JSON-like content between square brackets
                start = cleaned_response.find('[')
                end = cleaned_response.rfind(']') + 1
                if start != -1 and end != 0:
                    json_part = cleaned_response[start:end]
                    mcqs = json.loads(json_part)
                    return mcqs
            except:
                pass
            
            # Fallback: Generate structured MCQs manually
            return self._generate_fallback_mcqs(topic, context)
            
        except Exception as e:
            print(f"Error generating MCQs: {e}")
            return self._generate_fallback_mcqs(topic, context)

    def _generate_fallback_mcqs(self, topic: str, context: str = "") -> List[Dict]:
        """Generate fallback MCQs when Gemini fails"""
        return [
            {
                "question": f"What is the primary focus of {topic}?",
                "options": [
                    "To replace human intelligence",
                    "To enhance human capabilities through technology",
                    "To eliminate the need for data",
                    "To make computers faster"
                ],
                "correct_answer": "B",
                "explanation": f"{topic} focuses on enhancing human capabilities through technological advancement."
            },
            {
                "question": f"Which of the following is a key component of {topic}?",
                "options": [
                    "Hardware only",
                    "Software only",
                    "Both hardware and software",
                    "Neither hardware nor software"
                ],
                "correct_answer": "C",
                "explanation": f"{topic} requires both hardware and software components to function effectively."
            },
            {
                "question": f"What is the main benefit of studying {topic}?",
                "options": [
                    "To become a programmer",
                    "To understand modern technology and its applications",
                    "To avoid using computers",
                    "To memorize facts"
                ],
                "correct_answer": "B",
                "explanation": f"Studying {topic} helps understand how modern technology works and its real-world applications."
            },
            {
                "question": f"Which field is most closely related to {topic}?",
                "options": [
                    "Literature",
                    "Mathematics and Computer Science",
                    "Art History",
                    "Physical Education"
                ],
                "correct_answer": "B",
                "explanation": f"{topic} is closely related to mathematics and computer science as it involves algorithms and computational thinking."
            },
            {
                "question": f"What skill is most important for {topic}?",
                "options": [
                    "Memorization",
                    "Critical thinking and problem-solving",
                    "Physical strength",
                    "Artistic ability"
                ],
                "correct_answer": "B",
                "explanation": f"Critical thinking and problem-solving are essential skills for understanding and working with {topic}."
            }
        ]
    
    def generate_summary(self, content: str) -> str:
        """Generate a summary of the provided content"""
        if not self.use_gemini:
            return f"Mock summary: {content[:200]}... (API key not configured)"
        
        prompt = f"""
        Please provide a comprehensive summary of the following content:
        
        {content}
        
        The summary should:
        1. Capture the main points and key concepts
        2. Be well-structured and easy to understand
        3. Include important details while being concise
        4. Be suitable for study purposes
        """
        
        return self.generate_response(prompt)
    
    def grade_answer(self, question: str, correct_answer: str, student_answer: str, context: str = "") -> Dict:
        """Grade a student's answer"""
        if not self.use_gemini:
            return {
                "score": 75,
                "feedback": "Mock feedback (API key not configured)",
                "strengths": ["Mock strength"],
                "improvements": ["Mock improvement"],
                "suggestions": ["Mock suggestion"]
            }
        
        prompt = f"""
        Grade the following student answer:
        
        Question: {question}
        Correct Answer: {correct_answer}
        Student Answer: {student_answer}
        Context: {context}
        
        Provide a detailed evaluation including:
        1. Score (0-100)
        2. Feedback on what was correct
        3. Areas for improvement
        4. Suggestions for better answers
        
        Format as JSON:
        {{
            "score": 85,
            "feedback": "Detailed feedback here",
            "strengths": ["What was good"],
            "improvements": ["What to improve"],
            "suggestions": ["How to improve"]
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            grade_data = json.loads(response)
            return grade_data
        except json.JSONDecodeError:
            return {
                "score": 0,
                "feedback": "Error processing grade",
                "strengths": [],
                "improvements": [],
                "suggestions": []
            }