from flask import Flask, request, jsonify, Response
import requests
import psutil
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import quote
import re

app = Flask(__name__)
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:3b"
HTML_FILE = Path(__file__).with_name("JARVIS_Futuristic_Website.html")

NEWS_WORDS = re.compile(r"\b(today|latest|news|current|india|bharat)\b|भारत|आज|ताज़ा|ताजा|खबर|समाचार|न्यूज़|न्यूज", re.I)

def get_news(query):
    """Fetch current headlines from Google News RSS and preserve article links."""
    q = query.strip() or "India latest news"
    if not NEWS_WORDS.search(q):
        return ""
    url = "https://news.google.com/rss/search?q=" + quote(q) + "&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 JARVIS local assistant"}
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall(".//item")[:6]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title and link:
                items.append(f"- {title}\n  Published: {pub}\n  Link: {link}")
        return "\n".join(items)
    except Exception:
        return ""

@app.get("/")
def home():
    if not HTML_FILE.exists():
        return "JARVIS_Futuristic_Website.html is missing. Put it beside jarvis_web.py.", 404
    return Response(HTML_FILE.read_text(encoding="utf-8"), mimetype="text/html")

@app.get("/api/status")
def status():
    vm = psutil.virtual_memory()
    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.1),
        "memory_used": round((vm.total-vm.available)/(1024**3), 1),
        "memory_total": round(vm.total/(1024**3), 1),
        "model": MODEL
    })

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or data.get("prompt") or "").strip()
    if not message:
        return jsonify({"error": "Please type or speak a question."}), 400

    news = get_news(message)
    system_prompt = (
        "You are JARVIS, a helpful personal assistant. Reply in the user's language. "
        "Be concise, clear, and accurate. For current news, use only the supplied search results, "
        "include article titles and URLs, and do not invent facts or links. If no search results "
        "were retrieved, clearly say live search did not return results."
    )
    prompt = message
    if NEWS_WORDS.search(message):
        if news:
            prompt += "\n\nLive Google News RSS results:\n" + news
        else:
            prompt += "\n\nLive news search returned no results. Do not invent current news; say that the search failed."
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 220}
            },
            timeout=180
        )
        response.raise_for_status()
        result = response.json()
        reply = (result.get("message") or {}).get("content", "").strip()
        if not reply:
            return jsonify({"error": "Ollama returned an empty answer."}), 502
        return jsonify({"reply": reply, "sources_found": bool(news)})
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama is not running. Please ensure Ollama is installed and running."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "AI took too long. Try a shorter question."}), 504
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": "Ollama error: " + str(exc)[:180]}), 502

if __name__ == "__main__":
    print("JARVIS server starting. Keep this window open.")
    print("Open http://127.0.0.1:5000 in Chrome.")
    print("Model:", MODEL)
    app.run(host="127.0.0.1", port=5000, debug=False)
