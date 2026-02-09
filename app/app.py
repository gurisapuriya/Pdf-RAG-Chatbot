import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import tempfile

st.set_page_config(page_title="Dynamic PDF RAG Chatbot", page_icon="📚")
st.title("📚 Dynamic PDF Q&A Chatbot (Local Ollama RAG)")
st.write("Upload PDFs → 'Process PDFs' → Ask questions! Like ChatPDF, but fully local/private with citations.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write(f"{len(uploaded_files)} PDF(s) uploaded: {', '.join(f.name for f in uploaded_files)}")

    if st.button("Process PDFs & Start Chat"):
        with st.spinner("Processing (extraction, chunking, indexing — 1-2 mins)..."):
            docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                loader = UnstructuredPDFLoader(tmp_path, mode="elements")
                elements = loader.load()
                # Add filename to metadata for citations
                for elem in elements:
                    elem.metadata["source"] = uploaded_file.name
                chunks = text_splitter.split_documents(elements)
                docs.extend(chunks)

                os.unlink(tmp_path)

            docs = filter_complex_metadata(docs)

            embedding = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)

            texts = [doc.page_content for doc in docs]
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            keyword_retriever = BM25Retriever.from_texts(texts)
            keyword_retriever.k = 10

            retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, keyword_retriever],
                weights=[0.6, 0.4]
            )

            template = """
            You are a helpful assistant. Answer using ONLY the uploaded PDFs.
            Quote direct text for facts. If unsure, say "I don't know."
            Cite sources at end.

            Context:
            {context}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                formatted = []
                for doc in docs:
                    content = doc.page_content.strip()
                    source = doc.metadata.get("source", "uploaded PDF")
                    page = doc.metadata.get("page")
                    if page is not None:
                        page = page + 1  # 1-indexed
                    else:
                        page = "unknown"
                    formatted.append(f"{content}\n(Source: {source}, page {page})")
                return "\n\n".join(formatted)

            llm = ChatOllama(model="llama3", temperature=0.0)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.chain = chain

        st.success("PDFs processed! Ask questions below.")

if st.session_state.chain:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Ask about the uploaded PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving & generating..."):
                response = st.session_state.chain.invoke(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Upload PDFs and process to start.")

st.sidebar.header("Features")
st.sidebar.write("- Dynamic upload (any PDFs)")
st.sidebar.write("- Local Ollama Llama3 + nomic embeddings")
st.sidebar.write("- Hybrid retrieval + citations")
st.sidebar.write("- Built by Gurleen Singh")