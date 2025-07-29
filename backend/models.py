from pydantic import BaseModel
from typing import List, Dict, Any

class AnalyzeRequest(BaseModel):
    keywords: List[str]

class AnalyzeResponse(BaseModel):
    results: Dict[str, Any]
