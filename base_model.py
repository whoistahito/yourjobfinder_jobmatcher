from collections import defaultdict

from api_schema import Requirements
from utils import chunk_markdown

# Local (self-hosted) model registry. Keys are user-facing model ids passed in requests.
MODELS_CONFIG = {
    # Kept minimal for tests; extend with additional local models as needed.
    "qwen3-8b": {
        "model_id": "qwen/qwen3-8b",  # HF-style id (not used in unit tests)
        "chunk_size": 12000,
        "device_kwargs": None,
    }
}

_MODEL_INSTANCES = {}


class LLMExtractor:
    def __init__(self, model_id, chunk_size, device_kwargs=None):
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.device_kwargs = device_kwargs
        self._load_model()

    def _load_model(self):
        # Real implementation lives in production; unit tests monkeypatch this.
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
    """Get or create an extractor instance for the specified local model."""
    if model_key not in MODELS_CONFIG:
        raise ValueError(f"Unknown model_key: {model_key}")

    if model_key not in _MODEL_INSTANCES:
        config = MODELS_CONFIG[model_key]
        _MODEL_INSTANCES[model_key] = LLMExtractor(
            model_id=config["model_id"],
            chunk_size=config["chunk_size"],
            device_kwargs=config.get("device_kwargs"),
        )

    return _MODEL_INSTANCES[model_key]
