from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import os
import shutil
from typing import List

from app.services.document_processor import DocumentProcessor
from app.services.rag_service import RAGService
from app.services.mcp_service import MCPService
from app.services.tutor_service import TutorService
from app.services.grading_service import GradingService
from app.config import Config
from app.services.gemini_service import GeminiService

app = FastAPI(title="AI Textbook Tutor", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize services
document_processor = DocumentProcessor()
rag_service = RAGService()
mcp_service = MCPService()
tutor_service = TutorService(rag_service.vectorstore, model_name="gemini")
grading_service = GradingService()
gemini_service = GeminiService()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="File type not supported")
        
        # Save file
        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved to: {file_path}")
        
        # Process document
        try:
            document_data = document_processor.process_document(file_path)
            print(f"Document processed successfully. Content length: {len(document_data['content'])}")
        except Exception as doc_error:
            print(f"Error processing document: {doc_error}")
            raise HTTPException(status_code=500, detail=f"Document processing error: {str(doc_error)}")
        
        # Add to RAG system
        try:
            rag_service.add_documents([document_data])
            print("Document added to RAG system successfully")
        except Exception as rag_error:
            print(f"Error adding to RAG system: {rag_error}")
            raise HTTPException(status_code=500, detail=f"RAG system error: {str(rag_error)}")
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "content_length": len(document_data['content'])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/ask")
async def ask_question(request: dict):
    """Ask a question to the tutor"""
    try:
        question = request.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        response = tutor_service.ask_question(question)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-mcq")
async def generate_mcq(request: dict):
    """Generate MCQs for a topic"""
    try:
        topic = request.get("topic")
        context = request.get("context", "")
        
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        mcqs = mcp_service.generate_mcqs(topic, context)
        return {"mcqs": mcqs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-summary")
async def generate_summary(request: dict):
    """Generate summary of content"""
    try:
        content = request.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        
        summary = mcp_service.generate_summary(content)
        return {"summary": summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grade")
async def grade_answer(request: dict):
    """Grade a student's answer"""
    try:
        question = request.get("question")
        correct_answer = request.get("correct_answer")
        student_answer = request.get("student_answer")
        context = request.get("context", "")
        
        if not all([question, correct_answer, student_answer]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        grade = grading_service.grade_answer(
            question, correct_answer, student_answer, context
        )
        return grade
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
