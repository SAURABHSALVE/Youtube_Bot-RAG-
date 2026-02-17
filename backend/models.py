from pydantic import BaseModel
from typing import Optional, List

class VideoRequest(BaseModel):
    video_id: str
    language: Optional[str] = "en"

class QuestionRequest(BaseModel):
    video_id: str
    question: str
    language: Optional[str] = "en"

class AnswerResponse(BaseModel):
    answer: Optional[str] = None
    sources: Optional[List[str]] = []
    error: Optional[str] = None
