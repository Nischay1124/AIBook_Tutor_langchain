import os 
from dotenv import load_dotenv

load_dotenv()   

class Config:
    # API Configuration
    #OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    #OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tutor.db")
    
    # File Upload Configuration
    UPLOAD_FOLDER = "static/uploads"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.jpg', '.png', '.jpeg'}
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY = "data/embeddings"
    
    # Session Configuration
    SESSION_TIMEOUT = 3600