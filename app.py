from contextlib import asynccontextmanager
import os

from fastapi import Depends
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api_schema import JobExtractionInput, JobMatchingResponse
from external_model import get_extractor_for
from similarity_search import compute_similarity
from utils import merge_requirements


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        from similarity_search import get_model

        get_model()
    except Exception:
        pass
    yield


app = FastAPI(lifespan=lifespan)

_security = HTTPBearer(auto_error=False)


def _require_token(credentials: HTTPAuthorizationCredentials | None = Depends(_security)):
    """Require an Authorization: Bearer <token> header matching API_ACCESS_TOKEN.

    If API_ACCESS_TOKEN is unset/empty, auth is disabled (useful for local/dev).
    """
    expected = os.getenv("API_ACCESS_TOKEN", "")
    if not expected:
        return

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    if credentials.credentials != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/extract", response_model=JobMatchingResponse, dependencies=[Depends(_require_token)])
async def extract_requirements(input: JobExtractionInput,
                               response: Response,
                               ):
    try:
        if input.extractionPipeline is not None:
            extractor_ids = input.extractionPipeline.extractorModelIds
            if len(extractor_ids) < 2:
                raise ValueError("extractionPipeline.extractorModelIds must contain at least 2 items")

            req_a = get_extractor_for(extractor_ids[0]).process_text(input.inputText)
            req_b = get_extractor_for(extractor_ids[1]).process_text(input.inputText)
            merged = merge_requirements(req_a, req_b)

            judge = get_extractor_for(input.extractionPipeline.judgeModelId)
            requirements = judge.judge_requirements(input.inputText, merged)
        else:
            model = get_extractor_for(input.modelId)
            requirements = model.process_text(input.inputText)

        score = compute_similarity(input.userProfile, requirements)
        response.status_code = status.HTTP_200_OK
        return JobMatchingResponse(jobRequirements=requirements,
                                   userProfile=input.userProfile,
                                   similarityScore=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
