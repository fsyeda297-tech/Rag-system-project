import fitz
import os
from groq import Groq
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq and Pinecone initialized")


def load_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    print(f"✅ Loaded PDF ({len(doc)} pages)")
    doc.close()
    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": " ".join(chunk_words)
        })
        chunk_id += 1
        i += chunk_size - overlap
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def get_embedding(text):
    # Use Groq's embedding via a simple hash-based vector (384 dims)
    # We encode text to numbers using ord values
    import hashlib
    import struct
    
    words = text.lower().split()[:100]
    vector = [0.0] * 384
    
    for i, word in enumerate(words):
        hash_val = hashlib.md5(word.encode()).digest()
        for j in range(0, min(16, 384 - (i * 16)), 1):
            if i * 16 + j < 384:
                vector[i * 16 + j] = struct.unpack('b', bytes([hash_val[j]]))[0] / 128.0
    
    # Normalize
    magnitude = sum(x**2 for x in vector) ** 0.5
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    
    return vector


def upload_to_pinecone(chunks):
    print("⏳ Uploading to Pinecone...")
    vectors = []
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {"text": chunk["text"]}
        })

    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"✅ Uploaded batch {i//batch_size + 1}")

    print(f"🎉 All {len(vectors)} chunks uploaded!")


if __name__ == "__main__":
    pdf_path = "docs/Legal-Services-Agreement.pdf"

    if os.path.exists(pdf_path):
        text = load_pdf(pdf_path)
        chunks = chunk_text(text)
        upload_to_pinecone(chunks)
    else:
        print(f"❌ File not found: {pdf_path}")