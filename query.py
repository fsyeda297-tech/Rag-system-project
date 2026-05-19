import os
import hashlib
import struct
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
print("✅ System Ready!\n")


def get_embedding(text):
    words = text.lower().split()[:100]
    vector = [0.0] * 384
    for i, word in enumerate(words):
        hash_val = hashlib.md5(word.encode()).digest()
        for j in range(0, min(16, 384 - (i * 16)), 1):
            if i * 16 + j < 384:
                vector[i * 16 + j] = struct.unpack('b', bytes([hash_val[j]]))[0] / 128.0
    magnitude = sum(x**2 for x in vector) ** 0.5
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    return vector


def search_documents(query, top_k=3):
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches


def ask_groq(question, context_chunks):
    context = ""
    for i, match in enumerate(context_chunks):
        context += f"\n--- Source {i+1} ---\n"
        context += match.metadata.get("text", "")

    prompt = f"""You are a legal document assistant.
Answer the question based ONLY on the provided legal document context.
Always mention which source you used.
If the answer is not in the context, say "This information is not found in the document."

Context from Legal Document:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.1
    )
    return response.choices[0].message.content


def main():
    print("=" * 50)
    print("  Legal Document Assistant")
    print("  Ask questions about your legal document")
    print("  Type 'exit' to quit")
    print("=" * 50)
    print()

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        print("\n🔍 Searching document...")
        matches = search_documents(question)

        if not matches:
            print("❌ No relevant content found\n")
            continue

        print("🤖 Generating answer...\n")
        answer = ask_groq(question, matches)
        print(f"Assistant: {answer}")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()