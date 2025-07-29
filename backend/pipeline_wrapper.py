from main_pipeline import ContentIntelligencePipeline
from typing import List, Dict

# Simple wrapper to run the pipeline. In production this would include
# proper async management and error handling.

def run_pipeline(keywords: List[str]) -> Dict:
    pipeline = ContentIntelligencePipeline()
    return pipeline.run_full_pipeline(keywords)
