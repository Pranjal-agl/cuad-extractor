import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tiny_model_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("CUAD_MODEL_PATH", str(tiny_model_dir))
    monkeypatch.setenv("CUAD_LOG_PREDICTIONS", "1")
    from src.eval import monitor

    monkeypatch.setattr(monitor, "LOG_PATH", tmp_path / "production_log.jsonl")
    from src.serving.serve import app

    with TestClient(app) as c:
        yield c


def test_extract_returns_all_clause_types(client):
    r = client.post("/extract", json={"text": "This Agreement is governed by the laws of Delaware."})
    assert r.status_code == 200
    body = r.json()
    assert body["error"] is None
    assert set(body["clauses"]) == set(client.get("/clauses").json()["clause_types"])
    assert body["n_windows"] == 1
    for clause, spans in body["spans"].items():
        assert [s["text"] for s in spans] == body["clauses"][clause]


def test_extract_validation(client):
    assert client.post("/extract", json={"text": "   "}).status_code == 400
    r = client.post("/extract", json={"text": "x", "clause_types": ["Indemnification"]})
    assert r.status_code == 400
    assert client.post("/extract", json={"text": "x", "min_score": 2}).status_code == 422


def test_min_score_filters_everything(client):
    body = client.post("/extract", json={"text": "the laws of delaware " * 20, "min_score": 1.0}).json()
    assert all(v == [] for v in body["clauses"].values())


def test_health_metrics_and_prediction_log(client, tmp_path):
    client.post("/extract", json={"text": "the laws of delaware", "clause_types": ["Governing Law"]})
    h = client.get("/health").json()
    assert h["model_loaded"] and h["requests_tracked"] >= 1
    assert client.get("/metrics").json()["n_requests"] >= 1
    assert (tmp_path / "production_log.jsonl").read_text().strip()
