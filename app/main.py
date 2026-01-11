# app/main.py
from __future__ import annotations

import os
from flask import Flask, jsonify, request

from .rag import answer_question

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "Missing 'question'"}), 400

    result = answer_question(q)
    return jsonify(result)

@app.get("/")
def root():
    # Minimal landing page for now (HTML UI comes next)
    return (
        "<h2>RAG Chatbot is running</h2>"
        "<p>POST a JSON body to <code>/chat</code> like:</p>"
        "<pre>{\"question\": \"What is model validation?\"}</pre>"
        "<p>Health: <code>/health</code></p>"
    )

if __name__ == "__main__":
    # Local dev only; Render will use gunicorn
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

