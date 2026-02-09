import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# 1. Load embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# 2. Load existing Chroma vector DB (relative path)
db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
vectorstore = Chroma(
    persist_directory=db_path,
    embedding_function=embedding_model
)

# Load texts for BM25 keyword search
all_splits = vectorstore.get(include=["documents"])
texts = all_splits["documents"]

# 3. Retrievers
semantic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

keyword_retriever = BM25Retriever.from_texts(texts)  #from_texts avoids id/metadata issues
keyword_retriever.k = 10

# Hybrid retriever (semantic main + keyword boost for exact terms)
retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.7, 0.3]
)

# 4. Prompt template
template = """
You are an expert assistant. Answer the question using ONLY the provided context.
Extract direct quotes when possible. If no clear answer, say "I don't know."
Cite sources at the end of relevant sentences.

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 5. Format docs with sources/pages (handles keyword results without metadata)
def format_docs(docs):
    formatted = []
    for doc in docs:
        content = doc.page_content.strip()
        if hasattr(doc, "metadata") and doc.metadata:
            source = doc.metadata.get("source", "unknown").split("/")[-1]
            page = doc.metadata.get("page", 0) + 1
        else:
            source = "unknown (keyword match)"
            page = "unknown"
        formatted.append(f"{content}\n(Source: {source}, page {page})")
    return "\n\n".join(formatted)

# 6. LLM
llm = ChatOllama(model="llama3", temperature=0.1)  # Low temp for accuracy

# 7. RAG chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)