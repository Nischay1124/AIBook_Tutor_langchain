from typing import Dict, List
from app.services.gemini_service import GeminiService
import json

class GradingService:
    def __init__(self):
        self.gemini_service = GeminiService()
    
    def grade_answer(self, question: str, correct_answer: str, student_answer: str, context: str = "") -> Dict:
        """Grade student answer using Gemini"""
        return self.gemini_service.grade_answer(question, correct_answer, student_answer, context)
    
    def grade_mcq_answers(self, answers: List[Dict]) -> Dict:
        """Grade multiple choice question answers"""
        total_score = 0
        feedback = []
        
        for answer in answers:
            if answer['student_answer'] == answer['correct_answer']:
                total_score += 1
                feedback.append(f"Question {answer.get('question_id', 'Unknown')}: Correct!")
            else:
                feedback.append(
                    f"Question {answer.get('question_id', 'Unknown')}: Incorrect. "
                    f"Correct answer was {answer['correct_answer']}"
                )
        
        percentage = (total_score / len(answers)) * 100 if answers else 0
        
        return {
            'total_score': total_score,
            'total_questions': len(answers),
            'percentage': percentage,
            'feedback': feedback,
            'grade': self._get_letter_grade(percentage)
        }
    
    def _get_letter_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade"""
        if percentage >= 90:
            return 'A+'
        elif percentage >= 80:
            return 'A'
        elif percentage >= 70:
            return 'B'
        elif percentage >= 60:
            return 'C'
        elif percentage >= 50:
            return 'D'
        else:
            return 'F'
    
    def _fallback_grading(self, question: str, correct_answer: str, 
                         student_answer: str) -> Dict:
        """Fallback grading method"""
        return {
            'score': 5,
            'feedback': 'Unable to grade automatically. Please review manually.',
            'strengths': [],
            'improvements': [],
            'explanation': correct_answer
        }