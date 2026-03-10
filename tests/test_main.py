# tests/test_main.py
"""Unit tests for app/main.py Flask endpoints."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Return a Flask test client with testing mode enabled."""
    from app.main import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_ANSWER = {
    "answer": "This is a test answer.",
    "contexts": [{"faiss_id": 0, "score": 0.9, "page": 1, "chunk_id": "c1", "text": "ctx text"}],
    "hits": [{"faiss_id": 0, "score": 0.9, "page": 1, "chunk_id": "c1", "text": "ctx text"}],
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_json_status_ok(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data == {"status": "ok"}

    def test_health_content_type_is_json(self, client):
        resp = client.get("/health")
        assert resp.content_type.startswith("application/json")


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_html(self, client):
        resp = client.get("/")
        body = resp.data.decode()
        assert "<h2>" in body

    def test_root_mentions_chat_endpoint(self, client):
        resp = client.get("/")
        body = resp.data.decode()
        assert "/chat" in body


# ---------------------------------------------------------------------------
# /chat  –  method validation
# ---------------------------------------------------------------------------

class TestChatMethodValidation:
    def test_get_not_allowed(self, client):
        resp = client.get("/chat")
        assert resp.status_code == 405

    def test_put_not_allowed(self, client):
        resp = client.put("/chat", json={"question": "hi"})
        assert resp.status_code == 405


# ---------------------------------------------------------------------------
# /chat  –  request validation
# ---------------------------------------------------------------------------

class TestChatRequestValidation:
    def test_missing_json_body_returns_400(self, client):
        resp = client.post("/chat", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_question_field_returns_400(self, client):
        resp = client.post("/chat", json={"other": "value"})
        assert resp.status_code == 400

    def test_empty_question_returns_400(self, client):
        resp = client.post("/chat", json={"question": ""})
        assert resp.status_code == 400

    def test_whitespace_only_question_returns_400(self, client):
        resp = client.post("/chat", json={"question": "   "})
        assert resp.status_code == 400

    def test_error_body_contains_message(self, client):
        resp = client.post("/chat", json={"question": ""})
        data = resp.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# /chat  –  successful request
# ---------------------------------------------------------------------------

class TestChatSuccess:
    @patch("app.main.answer_question", return_value=MOCK_ANSWER)
    def test_valid_question_returns_200(self, mock_aq, client):
        resp = client.post("/chat", json={"question": "What is model validation?"})
        assert resp.status_code == 200

    @patch("app.main.answer_question", return_value=MOCK_ANSWER)
    def test_response_contains_answer_key(self, mock_aq, client):
        resp = client.post("/chat", json={"question": "What is model validation?"})
        data = resp.get_json()
        assert "answer" in data

    @patch("app.main.answer_question", return_value=MOCK_ANSWER)
    def test_response_contains_contexts_and_hits(self, mock_aq, client):
        resp = client.post("/chat", json={"question": "What is model validation?"})
        data = resp.get_json()
        assert "contexts" in data
        assert "hits" in data

    @patch("app.main.answer_question", return_value=MOCK_ANSWER)
    def test_response_content_type_is_json(self, mock_aq, client):
        resp = client.post("/chat", json={"question": "What is model validation?"})
        assert resp.content_type.startswith("application/json")

    @patch("app.main.answer_question", return_value=MOCK_ANSWER)
    def test_whitespace_stripped_before_dispatch(self, mock_aq, client):
        resp = client.post("/chat", json={"question": "  hello  "})
        assert resp.status_code == 200
        mock_aq.assert_called_once_with("hello")
