import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# Text splitter from langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document
from config import driver, embeddings, llm

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
    print("🔧 Building vector index...")
    return FAISS.from_documents(chunks, embeddings)

# ---------------------------
# 4️⃣ Graph + Vector RAG query
# ---------------------------
def graph_rag_query(question, topic="Neo4j"):
    with driver.session() as session:
        # Graph retrieval: get all chunks related to topic
        result = session.run("""
            MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
            WHERE d.name CONTAINS $topic
            RETURN c.text AS text
            LIMIT 5
        """, topic=topic)
        graph_context = "\n".join([r["text"] for r in result])

    # Vector retrieval
    vector_docs = vectorstore.similarity_search(question, k=3)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    # Combine both
    context = f"GRAPH CONTEXT:\n{graph_context}\n\nVECTOR CONTEXT:\n{vector_context}"

    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant answering based on research papers."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ])
    return response.content

# ---------------------------
# 🚀 5️⃣ Main pipeline
# ---------------------------
if __name__ == "__main__":
    docs = load_documents("uploads")
    chunks = split_documents(docs)
    store_in_neo4j(chunks)
    vectorstore = build_vectorstore(chunks)

    query = "How does Neo4j improve retrieval in RAG systems?"
    answer = graph_rag_query(query, topic="Neo4j")

    print("\n💬 Answer:")
    print(answer)

    driver.close()