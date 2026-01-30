# test_api.py
import pytest
from fastapi.testclient import TestClient

# Change "app" to your filename (without .py) if needed, e.g. "main"
import app as app_module


@pytest.fixture()
def client(monkeypatch):
    """
    Creates a TestClient with all heavy dependencies mocked out.
    """

    # --- Mock LABELS mapping (if your config.LABELS is used) ---
    monkeypatch.setattr(app_module, "LABELS", {0: "negative", 1: "neutral", 2: "positive"}, raising=False)

    # --- Mock segmentation ---
    def fake_segment_text_all(text: str):
        # Just return a single "segmented" version for tests
        return ["this is a pen"] if text == "thisisapen" else [text]

    monkeypatch.setattr(app_module, "segment_text_all", fake_segment_text_all, raising=True)

    # --- Mock split_with_offsets ---
    def fake_split_with_offsets(text: str):
        # Return 2 spans for predictable testing
        return [
            {"start": 0, "end": min(10, len(text)), "text": text[: min(10, len(text))]},
            {"start": min(10, len(text)), "end": len(text), "text": text[min(10, len(text)) :]},
        ]

    monkeypatch.setattr(app_module, "split_with_offsets", fake_split_with_offsets, raising=True)

    # --- Mock classifier ---
    class FakeClassifier:
        def __call__(self, text: str):
            # Simple deterministic mapping for tests
            t = (text or "").lower()
            if "bad" in t or "awful" in t:
                return [{"label": "LABEL_0", "score": 0.9}]
            if "okay" in t or "meh" in t:
                return [{"label": "LABEL_1", "score": 0.7}]
            return [{"label": "LABEL_2", "score": 0.95}]

    fake_classifier = FakeClassifier()

    # get_classifier returns our fake classifier
    monkeypatch.setattr(app_module, "get_classifier", lambda: fake_classifier, raising=True)

    # is_model_loaded for /health
    monkeypatch.setattr(app_module, "is_model_loaded", lambda: True, raising=True)

    # load_model shouldn't do anything during tests
    monkeypatch.setattr(app_module, "load_model", lambda: None, raising=True)

    return TestClient(app_module.app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["model_loaded"] is True


def test_segment_empty(client):
    res = client.post("/segment", json={"text": ""})
    assert res.status_code == 200
    assert res.json()["error"] == "Empty text"


def test_segment_ok(client):
    res = client.post("/segment", json={"text": "thisisapen"})
    assert res.status_code == 200
    data = res.json()
    assert data["original"] == "thisisapen"
    assert data["segmentations"] == ["this is a pen"]
    assert data["count"] == 1


def test_predict_empty(client):
    res = client.post("/predict", json={"text": "   "})
    assert res.status_code == 200
    assert res.json()["error"] == "Empty text"


def test_predict_positive(client):
    res = client.post("/predict", json={"text": "The food was amazing!"})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_negative(client):
    res = client.post("/predict", json={"text": "This was awful and bad."})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "negative"
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_spans_ok(client):
    text = "Good food. Bad service."
    res = client.post("/predict_spans", json={"text": text})
    assert res.status_code == 200
    data = res.json()

    assert "overall" in data
    assert "label" in data["overall"]
    assert "confidence" in data["overall"]
    assert 0.0 <= data["overall"]["confidence"] <= 1.0

    assert "spans" in data
    assert isinstance(data["spans"], list)
    assert len(data["spans"]) == 2

    # Ensure each span has required keys
    for s in data["spans"]:
        assert set(["start", "end", "text", "label", "confidence"]).issubset(s.keys())
        assert 0.0 <= s["confidence"] <= 1.0


def test_predict_model_not_loaded(monkeypatch):
    # Make get_classifier return None for this test
    monkeypatch.setattr(app_module, "get_classifier", lambda: None, raising=True)

    client = TestClient(app_module.app)
    res = client.post("/predict", json={"text": "Hello"})
    assert res.status_code == 200
    assert res.json()["error"] == "Model not loaded"
