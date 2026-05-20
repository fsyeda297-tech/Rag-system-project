import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-rag")

TEMPLATE = """You are a legal assistant. Answer ONLY from context below.
If not found say: This information is not found in the provided documents.

Context: {context}

Question: {question}

Answer:"""

LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=TEMPLATE
)

def ask_legal_question(question: str) -> dict:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = LEGAL_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt).content
    sources = []
    for doc in docs:
        sources.append({
            "doc_name": doc.metadata.get("doc_name", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "excerpt": doc.page_content[:200] + "..."
        })
    return {"question": question, "answer": answer, "sources": sources}
