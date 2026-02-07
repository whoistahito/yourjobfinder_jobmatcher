from fastapi.testclient import TestClient


def test_extract_endpoint_happy_path(monkeypatch):
    import os
    os.environ["API_ACCESS_TOKEN"] = "testtoken"

    import app as app_module
    from api_schema import Requirements, SimilarityScore

    class StubExtractor:
        def process_text(self, text):
            return Requirements(skills=["python"], experiences=[], qualifications=[])

    monkeypatch.setattr(app_module, "get_extractor_for", lambda model_id: StubExtractor())
    monkeypatch.setattr(app_module, "compute_similarity", lambda user, req: SimilarityScore(score=0.5))

    client = TestClient(app_module.app)

    payload = {
        "modelId": "qwen3-8b",
        "inputText": "text",
        "userProfile": {"skills": ["python"], "experiences": [], "qualifications": []},
    }
    res = client.post("/extract", json=payload, headers={"Authorization": "Bearer testtoken"})

    assert res.status_code == 200
    body = res.json()
    assert body["jobRequirements"]["skills"] == ["python"]
    assert body["similarityScore"]["score"] == 0.5


def test_extract_endpoint_requires_token_when_configured(monkeypatch):
    import os
    os.environ["API_ACCESS_TOKEN"] = "testtoken"

    import app as app_module

    client = TestClient(app_module.app)
    payload = {
        "modelId": "qwen3-8b",
        "inputText": "text",
        "userProfile": {"skills": [], "experiences": [], "qualifications": []},
    }

    res = client.post("/extract", json=payload)
    assert res.status_code == 401

    res = client.post("/extract", json=payload, headers={"Authorization": "Bearer wrong"})
    assert res.status_code == 403


def test_extract_endpoint_returns_500_on_exception(monkeypatch):
    import os
    os.environ["API_ACCESS_TOKEN"] = "testtoken"

    import app as app_module

    def boom(_):
        raise RuntimeError("bad")

    monkeypatch.setattr(app_module, "get_extractor_for", boom)

    client = TestClient(app_module.app)
    payload = {
        "modelId": "qwen3-8b",
        "inputText": "text",
        "userProfile": {"skills": [], "experiences": [], "qualifications": []},
    }
    res = client.post("/extract", json=payload, headers={"Authorization": "Bearer testtoken"})
    assert res.status_code == 500
    assert "bad" in res.json()["detail"]


def test_extract_endpoint_ensemble_pipeline(monkeypatch):
    import os
    os.environ["API_ACCESS_TOKEN"] = "testtoken"

    import app as app_module
    from api_schema import Requirements, SimilarityScore

    calls = []

    class ExtractorA:
        def process_text(self, text):
            calls.append(("a", text))
            return Requirements(skills=["python"], experiences=[], qualifications=[])

    class ExtractorB:
        def process_text(self, text):
            calls.append(("b", text))
            return Requirements(skills=["fastapi"], experiences=["3 years"], qualifications=[])

    class Judge:
        def process_text(self, text):
            raise AssertionError("judge should not be used for extraction")

        def judge_requirements(self, input_text, requirements):
            calls.append(("judge", input_text, requirements.model_dump()))
            return Requirements(
                skills=sorted(requirements.skills + ["docker"]),
                experiences=requirements.experiences,
                qualifications=requirements.qualifications,
            )

    def fake_get_extractor_for(model_id):
        if model_id == "gpt-oss-120b":
            return ExtractorA() if not any(c[0] == "a" for c in calls) else Judge()
        if model_id == "qwen3-next-80b-thinking":
            return ExtractorB()
        raise AssertionError(model_id)

    monkeypatch.setattr(app_module, "get_extractor_for", fake_get_extractor_for)

    def fake_similarity(user, req):
        assert "docker" in req.skills
        assert set(req.skills) == {"docker", "fastapi", "python"}
        return SimilarityScore(score=0.9)

    monkeypatch.setattr(app_module, "compute_similarity", fake_similarity)

    client = TestClient(app_module.app)
    payload = {
        "modelId": "qwen3-8b",
        "extractionPipeline": {
            "extractorModelIds": ["gpt-oss-120b", "qwen3-next-80b-thinking"],
            "judgeModelId": "gpt-oss-120b",
        },
        "inputText": "job text",
        "userProfile": {"skills": ["python"], "experiences": [], "qualifications": []},
    }

    res = client.post("/extract", json=payload, headers={"Authorization": "Bearer testtoken"})
    assert res.status_code == 200

    body = res.json()
    assert set(body["jobRequirements"]["skills"]) == {"docker", "fastapi", "python"}
    assert body["jobRequirements"]["experiences"] == ["3 years"]
    assert body["similarityScore"]["score"] == 0.9

    assert [c[0] for c in calls] == ["a", "b", "judge"]
