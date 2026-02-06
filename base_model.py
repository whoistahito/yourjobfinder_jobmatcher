from collections import defaultdict
from api_schema import Requirements
from utils import chunk_markdown

MODELS_CONFIG = {}
_MODEL_INSTANCES = {}

class LLMExtractor:
    def __init__(self, model_id, chunk_size, device_kwargs=None):
        self.model_id = model_id
        self.chunk_size = chunk_size
        pass

    def _load_model(self):
        pass

    def process_text(self, text):
        chunks = chunk_markdown(text, self.chunk_size)
        all_requirements = []
        for chunk in chunks:
            req = self.process_chunk(chunk)
            all_requirements.append(req)
        merged = defaultdict(set)
        for req in all_requirements:
            merged["skills"].update(req.skills)
            merged["experiences"].update(req.experiences)
            merged["qualifications"].update(req.qualifications)
        unique_requirements = {k: sorted(list(v)) for k, v in merged.items()}
        return Requirements(**unique_requirements)

    def process_chunk(self, chunk) -> Requirements:
        return Requirements()

def get_extractor_for(model_key):
    if model_key not in MODELS_CONFIG:
        return None
    return _MODEL_INSTANCES.get(model_key)
