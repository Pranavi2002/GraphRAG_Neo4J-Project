# no multi-hop traversal and delete docs 

import os
# Loaders and vectorstore from langchain_community
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

# Text splitter from langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import driver, embeddings, llm

# ---------------------------
# Global variable for vectorstore
# ---------------------------
vectorstore = None

# ---------------------------
# 1️⃣ Load and split documents
# ---------------------------
def load_documents(folder_path="uploads"):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} documents")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# ---------------------------
# 2️⃣ Store docs into Neo4j
# ---------------------------
def store_in_neo4j(chunks):
    with driver.session() as session:
        for idx, chunk in enumerate(chunks):
            session.run("""
                MERGE (d:Document {name: $doc_name})
                CREATE (c:Chunk {id: $id, text: $text})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, doc_name=chunk.metadata.get("source", "unknown"),
                 id=idx, text=chunk.page_content)
    print("✅ Stored chunks in Neo4j")

# ---------------------------
# 3️⃣ Build FAISS vector store
# ---------------------------
def build_vectorstore(chunks):
    global vectorstore
    print("🔧 Building vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ---------------------------
# 4️⃣ Graph + Vector RAG query
# ---------------------------
def graph_rag_query(question, topic="Neo4j", k_graph=5, k_vector=3, use_docs_only=True):
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore not built yet. Upload and process documents first!")

    # Graph retrieval
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
            WHERE d.name CONTAINS $topic
            RETURN c.text AS text
            LIMIT $limit
        """, topic=topic, limit=k_graph)
        graph_context = "\n".join([r["text"] for r in result])

    # Vector retrieval
    vector_docs = vectorstore.similarity_search(question, k=k_vector)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    # Combine contexts
    context = f"GRAPH CONTEXT:\n{graph_context}\n\nVECTOR CONTEXT:\n{vector_context}".strip()

    # If no relevant info found in documents
    if use_docs_only and not vector_context.strip() and not graph_context.strip():
        return (
            f"The uploaded documents do not contain information about '{question}'.\n"
            "💡 Uncheck to include general knowledge."
        )

    # System prompt based on checkbox
    if use_docs_only:
        system_prompt = (
            "You are a helpful assistant. Answer the user's question **using only the provided documents**. "
            "If the documents are limited, give as detailed an answer as possible based on what is available. "
            "Do not use general knowledge."
        )
    else:
        system_prompt = (
            "You are a helpful assistant answering based on research papers and general knowledge. "
            "Use document context if available, otherwise use your general knowledge."
        )

    # LLM call
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ])

    answer = response.content

    # Append hint line if using docs only
    if use_docs_only:
        answer += "\n💡 Uncheck to include general knowledge."

    return answer