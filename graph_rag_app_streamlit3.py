# no delete a doc feature

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import driver, embeddings, llm

vectorstore = None

# ---------------------------
# 1️⃣ Load & split documents
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
# 2️⃣ Store in Neo4j
# ---------------------------
def store_in_neo4j(chunks):
    import json, re
    with driver.session() as session:
        for idx, chunk in enumerate(chunks):
            text = chunk.page_content
            doc_name = chunk.metadata.get("source", "unknown")

            session.run("""
                MERGE (d:Document {name: $doc_name})
                CREATE (c:Chunk {id: $doc_name + '_' + toString($id), text: $text})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, doc_name=doc_name, id=idx, text=text)

            extraction_prompt = f"""
            You are an information extraction assistant.
            Extract factual relationships from the text below **only** in JSON format.
            Each triple should have 'subject', 'relation', and 'object'.

            Example output:
            [
              {{"subject": "Barack Obama", "relation": "born in", "object": "Honolulu"}},
              {{"subject": "Honolulu", "relation": "located in", "object": "United States"}}
            ]

            Text:
            \"\"\"{text}\"\"\"
            """

            try:
                response = llm.invoke([{"role": "user", "content": extraction_prompt}])
                triples_text = response.content.strip()
            except Exception as e:
                print(f"⚠️ LLM extraction failed for chunk {idx}: {e}")
                continue

            match = re.search(r'\[.*\]', triples_text, re.DOTALL)
            if not match:
                print(f"⚠️ No JSON found for chunk {idx}")
                continue

            try:
                triples = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error in chunk {idx}: {e}")
                continue

            for triple in triples:
                subj = triple.get("subject")
                rel = triple.get("relation", "RELATED_TO")
                obj = triple.get("object")

                if subj and obj:
                    rel = rel.upper()
                    rel = re.sub(r'[^A-Z0-9_]', '_', rel)

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
# 3️⃣ Build FAISS vectorstore
# ---------------------------
def build_vectorstore(chunks):
    global vectorstore
    print("🔧 Building vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ---------------------------
# 4️⃣ Graph + Vector RAG query (dynamic hops)
# ---------------------------
def graph_rag_query(question, topic="Neo4j", k_graph=5, k_vector=3, hops=3, use_docs_only=True):
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore not built yet. Upload and process documents first!")

    with driver.session() as session:
        # Dynamically inject the hops value into the Cypher query
        cypher_query = f"""
            MATCH path=(start:Entity)-[*1..{int(hops)}]-(related)
            WHERE ANY(node IN nodes(path) WHERE node.name CONTAINS $topic)
            WITH DISTINCT related
            MATCH (related)<-[:HAS_CHUNK]-(d:Document)
            RETURN d.name AS doc_name, collect(DISTINCT related.name) AS related_entities
            LIMIT $limit
        """

        result = session.run(cypher_query, topic=topic, limit=k_graph)

        graph_contexts = [
            f"{record['doc_name']} mentions: {', '.join(record['related_entities'])}"
            for record in result
        ]
        graph_context = "\n".join(graph_contexts)

    # Vector retrieval
    vector_docs = vectorstore.similarity_search(question, k=k_vector)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    context = f"GRAPH CONTEXT:\n{graph_context}\n\nVECTOR CONTEXT:\n{vector_context}".strip()

    # Handle doc-only restriction
    if use_docs_only and not graph_context.strip() and not vector_context.strip():
        return (
            f"⚠️ The uploaded documents do not contain information about '{question}'.\n"
            "💡 Uncheck 'Use only uploaded documents' to include general knowledge."
        )

    # LLM prompt
    system_prompt = (
        "You are a helpful assistant. "
        + ("Answer using **only the provided document context**."
           if use_docs_only
           else "Use the document context and general knowledge if needed.")
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ])

    answer = response.content
    if use_docs_only:
        answer += "\n💡 Uncheck to include general knowledge."

    return answer

# ---------------------------
# 5️⃣ Delete all docs & entities
# ---------------------------
def delete_all_docs():
    with driver.session() as session:
        try:
            session.run("MATCH ()-[r]->() DELETE r")
            session.run("MATCH (n) WHERE n:Document OR n:Chunk OR n:Entity DELETE n")
            print("🗑️ All uploaded documents, chunks, and entities deleted from Neo4j.")
        except Exception as e:
            print(f"⚠️ Error deleting data: {e}")