from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
from .models import GenerationRequest, GenerationResponse, HealthResponse
from .inference import ModelServer

app = FastAPI(title="Domain-Specific LLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_PATH = os.getenv("MODEL_PATH", "./models/finetuned/medical-llm")
model_server = ModelServer(model_path=MODEL_PATH)

@app.get("/", response_model=dict)
async def root():
    return {"message": "Domain-Specific LLM API", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", device=model_server.device)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        result = model_server.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return GenerationResponse(
            text=result["text"],
            tokens_generated=result["tokens_generated"],
            latency_ms=result["latency_ms"],
            context_used=request.use_rag
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

