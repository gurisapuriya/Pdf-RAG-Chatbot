import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

#loading our embedding model form huggingface
embedding_model = OllamaEmbeddings(model = "nomic-embed-text")

#chunking strategy used is text-splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap = 200)

#loading all the PDFs
docs = []
data_dir = "/Users/HP/Pdf-RAG-Chatbot/data"
for filename in os.listdir(data_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(data_dir, filename)
        print(f"loading {filename}..")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)
        docs.extend(chunks)
        print(f" -> {len(pages)} pages  -> {len(chunks)} chunks")

if not docs:
    raise ValueError("No PDFs found in /data folder! Add some and then re-run.")

#creating our vector DB
print("creating vector DB")
vectorstore = Chroma.from_documents(
    documents = docs, 
    embedding = embedding_model, 
    persist_directory = '/Users/HP/Pdf-RAG-Chatbot/chroma_db')

print(f"Ingestion is complete! {len(docs)} chunks indexed in Chroma DB.")
print("this script could be run anytime new PDFs are added.")