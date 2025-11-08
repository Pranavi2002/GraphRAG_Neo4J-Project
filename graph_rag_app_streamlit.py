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
    import json
    import re

    with driver.session() as session:
        for idx, chunk in enumerate(chunks):
            text = chunk.page_content
            doc_name = chunk.metadata.get("source", "unknown")

            # 🗂️ Store document and chunk
            session.run("""
                MERGE (d:Document {name: $doc_name})
                CREATE (c:Chunk {id: $id, text: $text})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, doc_name=doc_name, id=idx, text=text)

            # ------------- 🧠 ENTITY EXTRACTION -------------
            extraction_prompt = f"""
            You are an information extraction assistant.
            Extract factual relationships from the text below **only** in JSON format.
            Each triple should have 'subject', 'relation', and 'object'.
            Do NOT include explanations, commentary, or markdown formatting.

            Example output:
            [
              {{"subject": "Barack Obama", "relation": "born in", "object": "Honolulu"}},
              {{"subject": "Honolulu", "relation": "located in", "object": "United States"}}
            ]

            Text:
            \"\"\"{text}\"\"\"
            """

            try:
                response = llm.invoke([
                    {"role": "user", "content": extraction_prompt}
                ])
                triples_text = response.content.strip()
            except Exception as e:
                print(f"⚠️ LLM extraction failed for chunk {idx}: {e}")
                continue

            # -------- 🧹 Clean non-JSON wrappers --------
            match = re.search(r'\[.*\]', triples_text, re.DOTALL)
            if match:
                triples_text = match.group(0)
            else:
                print(f"⚠️ No JSON found for chunk {idx}")
                continue

            # Parse JSON safely
            try:
                triples = json.loads(triples_text)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error in chunk {idx}: {e}")
                continue

            # ------------- 🧩 STORE ENTITIES & RELATIONSHIPS -------------
            for triple in triples:
                subj = triple.get("subject")
                rel = triple.get("relation", "RELATED_TO")
                obj = triple.get("object")

                if subj and obj:
                    # 🧹 Clean and normalize relationship name
                    rel = rel.upper()
                    rel = re.sub(r'[^A-Z0-9_]', '_', rel)  # replace invalid chars

                    query = f"""
                        MERGE (s:Entity {{name: $subj}})
                        MERGE (o:Entity {{name: $obj}})
                        MERGE (s)-[r:{rel}]->(o)
                        MERGE (c:Chunk {{id: $id}})-[:MENTIONS]->(s)
                        MERGE (c)-[:MENTIONS]->(o)
                    """

                    try:
                        session.run(query, subj=subj, obj=obj, id=idx)
                    except Exception as e:
                        print(f"⚠️ Neo4j write error for relation '{rel}' in chunk {idx}: {e}")

        print("✅ Stored chunks and extracted entities in Neo4j")

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
def graph_rag_query(question, topic="Neo4j", k_graph=5, k_vector=3, hops=3, use_docs_only=True):
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore not built yet. Upload and process documents first!")

    # ---------------------------
    # 1️⃣ Graph context (multi-hop traversal)
    # ---------------------------
    with driver.session() as session:
        result = session.run(f"""
            MATCH path=(start:Entity)-[*1..{hops}]-(related)
            WHERE ANY(node IN nodes(path) WHERE node.name CONTAINS $topic)
            WITH DISTINCT related
            MATCH (related)<-[:HAS_CHUNK]-(d:Document)
            RETURN d.name AS doc_name, collect(DISTINCT related.name) AS related_entities
            LIMIT $limit
        """, topic=topic, limit=k_graph)

        graph_contexts = []
        for record in result:
            doc_name = record["doc_name"]
            entities = ", ".join(record["related_entities"])
            graph_contexts.append(f"{doc_name} mentions: {entities}")

        graph_context = "\n".join(graph_contexts)

    # ---------------------------
    # 2️⃣ Vector context
    # ---------------------------
    vector_docs = vectorstore.similarity_search(question, k=k_vector)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    # ---------------------------
    # 3️⃣ Combine context
    # ---------------------------
    context = f"GRAPH CONTEXT:\n{graph_context}\n\nVECTOR CONTEXT:\n{vector_context}".strip()

    # ---------------------------
    # 4️⃣ Docs-only check
    # ---------------------------
    if use_docs_only and not graph_context.strip() and not vector_context.strip():
        return (
            f"⚠️ The uploaded documents do not contain information about '{question}'.\n"
            "💡 Uncheck 'Use only uploaded documents' to include general knowledge."
        )

    # ---------------------------
    # 5️⃣ System prompt
    # ---------------------------
    if use_docs_only:
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using **only the provided document context**. "
            "Do not use any general knowledge beyond what is provided."
        )
    else:
        system_prompt = (
            "You are a helpful assistant. Use the provided document context to answer the question. "
            "If relevant info is missing, you may also use general knowledge."
        )

    # ---------------------------
    # 6️⃣ LLM call
    # ---------------------------
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ])

    answer = response.content
    if use_docs_only:
        answer += "\n💡 Uncheck to include general knowledge."

    return answer

# ---------------------------
# 5️⃣ Delete all docs & entities from Neo4j
# ---------------------------
def delete_all_docs():
    with driver.session() as session:
        try:
            # Delete relationships first
            session.run("MATCH ()-[r]->() DELETE r")
            # Then delete the relevant nodes
            session.run("MATCH (n) WHERE n:Document OR n:Chunk OR n:Entity DELETE n")
            print("🗑️ All uploaded documents, chunks, and entities deleted from Neo4j.")
        except Exception as e:
            print(f"⚠️ Error deleting data: {e}")