from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from models import VideoRequest, QuestionRequest, AnswerResponse
from rag import RAGProcessor
from guardrails import Guardrails
import uvicorn
import os

app = FastAPI()

# Allow CORS for extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGProcessor()
guards = Guardrails()

@app.get("/")
async def root():
    return {"status": "ok", "message": "YouTube RAG API is running"}

@app.post("/api/process", response_model=AnswerResponse)
async def process_video_endpoint(request: VideoRequest):
    try:
        # process_video is sync, but we want to run it in threadpool. 
        # With async def, we should use run_in_executor or asyncio.to_thread manually
        # Or if we defined this endpoint as 'def', FastAPI would do it.
        # Since we use 'async def' (good practice), let's offload.
        import asyncio
        await asyncio.to_thread(rag.process_video, request.video_id, request.language)
        return AnswerResponse(answer="Video processed successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question_endpoint(request: QuestionRequest):
    # Guardrails: Input validation
    is_valid, msg = guards.validate_input(request.question)
    if not is_valid:
        return AnswerResponse(error=msg)

    try:
        answer = await rag.answer_question(request.video_id, request.question, request.language)
        
        # Guardrails: Output validation
        is_valid_ans, msg_ans = guards.validate_answer(answer)
        if not is_valid_ans:
             return AnswerResponse(answer="I cannot answer this question due to safety guidelines.", error=msg_ans)

        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
