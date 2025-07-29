"""FastAPI wrapper for the Social Media Intelligence Tool."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import AnalyzeRequest, AnalyzeResponse
from .pipeline_wrapper import run_pipeline

app = FastAPI(title="SMIT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(data: AnalyzeRequest):
    try:
        results = run_pipeline(data.keywords)
        return AnalyzeResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
async def health():
    return {"status": "ok"}
