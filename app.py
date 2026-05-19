from flask import Flask, request, jsonify
import os, hashlib, struct
from groq import Groq
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_embedding(text):
    words = text.lower().split()[:100]
    vector = [0.0] * 384
    for i, word in enumerate(words):
        h = hashlib.md5(word.encode()).digest()
        for j in range(min(16, 384 - i*16)):
            if i*16+j < 384:
                vector[i*16+j] = struct.unpack("b", bytes([h[j]]))[0] / 128.0
    mag = sum(x**2 for x in vector)**0.5
    return [x/mag for x in vector] if mag > 0 else vector

@app.route("/")
def home():
    return open("templates/index.html", encoding="utf-8").read()

@app.route("/ask", methods=["POST"])
def ask():
    q = request.json.get("question", "")
    emb = get_embedding(q)
    res = index.query(vector=emb, top_k=3, include_metadata=True)
    ctx = " ".join([m.metadata.get("text","") for m in res.matches])
    prompt = f"Answer based on this legal document only:\n{ctx}\n\nQuestion: {q}\nAnswer:"
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        max_tokens=500
    )
    return jsonify({"answer": resp.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)
