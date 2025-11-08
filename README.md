# 📚 Graph + Vector RAG System (Neo4j-Focused)

This project demonstrates a **Retrieval-Augmented Generation (RAG) system** with a strong focus on **Neo4j graph database** for storing, retrieving, and querying document chunks and metadata. It integrates **FAISS** for vector similarity search and a **Streamlit frontend** for interacting with the system.

The system now supports:

* **Entity extraction** from documents using LLMs
* **Multi-hop graph traversal** in Neo4j to connect related entities
* **Vector similarity search** with FAISS
* Combined **graph + vector context** for improved RAG answers
* Option to **restrict answers to uploaded documents only**

---

## 📂 Folder Structure

```
graph_rag_project/
├── config.py                  # Neo4j credentials, embeddings, LLM (NOT included in GitHub)
├── graph_demo.py              # Neo4j demo: create graph nodes & relationships
├── graph_rag_app.py           # Backend RAG pipeline: load docs, split, store in Neo4j, build vectorstore
├── graph_rag_app_streamlit.py # Streamlit backend: Neo4j + vectorstore integration
├── streamlit_app.py           # Streamlit frontend UI
├── uploads/                   # PDFs/TXT files (ignored in GitHub)
```

---

## 🧩 Requirements

Python 3.10+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
streamlit==1.26.0
neo4j==5.14.0
langchain==0.1.218
langchain-community==0.0.30
faiss-cpu==1.7.4
```

---

## ⚙️ Neo4j Setup

1. **Install Neo4j Desktop**: [Download](https://neo4j.com/download/)

2. **Create Local Database**:

   * Add → Local DBMS
   * Name: `graph_rag_local`
   * Password: e.g., `neo4j123`
   * Version: Neo4j 5.x (default)

3. **Start Database**

   * Bolt URL: `bolt://localhost:7687`
   * Browser URL: `http://localhost:7474`

---

## 🏗️ Using Neo4j with Python

### 1️⃣ Create Graph (graph_demo.py)

* Nodes:

  * `Paper` → research papers
  * `Topic` → topics
  * `Author` → authors

* Relationships:

  * `RELATED_TO` → Paper ↔ Topic
  * `AUTHORED_BY` → Paper ↔ Author
  * `CONNECTED_TO` → Topic ↔ Topic

**Example Multi-Hop Query**:

```python
def multi_hop_query(tx, target_topic):
    query = """
    MATCH (p:Paper)-[:RELATED_TO]->(:Topic)-[:CONNECTED_TO]->(t:Topic {name: $target_topic})
    RETURN DISTINCT p.title AS title
    """
    result = tx.run(query, target_topic=target_topic)
    return [record["title"] for record in result]
```

Run:

```bash
python graph_demo.py
```

---

### 2️⃣ Neo4j Queries to Explore

* **View all nodes & relationships**:

```cypher
MATCH (n)-[r]->(m) RETURN n,r,m
```

* **View only nodes**:

```cypher
MATCH (n) RETURN n
```

* **View only relationships**:

```cypher
MATCH ()-[r]->() RETURN r
```

* **Delete all nodes & relationships**:

```cypher
MATCH (n) DETACH DELETE n
```

* **Remove duplicate relationships**:

```cypher
MATCH (a)-[r]->(b)
WITH a, b, TYPE(r) AS rel_type, COLLECT(r) AS rels
WHERE SIZE(rels) > 1
FOREACH (r IN TAIL(rels) | DELETE r)
```

---

### 3️⃣ Backend RAG Pipeline with Neo4j

* `graph_rag_app.py` and `graph_rag_app_streamlit.py`:

  * Load PDFs/TXT from `uploads/`
  * Split into chunks using `RecursiveCharacterTextSplitter`
  * Store chunks as `Chunk` nodes in Neo4j linked to `Document` nodes
  * Extract entities and relationships from chunks
  * Use **Neo4j graph** + **FAISS vector** retrieval
  * Combine both contexts for LLM-based answers

---

### 4️⃣ Streamlit Frontend

* Upload and process documents (Neo4j + FAISS)
* Ask questions about uploaded documents
* Option to **use only documents** or include general knowledge
* Clear uploaded documents with confirmation

```bash
streamlit run streamlit_app.py
```

---

## ✅ Docs-Only Mode

* When **“Use only uploaded documents”** is enabled:

  * The system **only uses graph + vector contexts from uploaded documents**
  * If a question is not covered, **no general-knowledge answer is provided**
  * Prevents hallucinations from unrelated external knowledge

* **Example**:

  1. Upload PDFs/TXT files.
  2. Enable the checkbox “Use only uploaded documents.”
  3. Ask a question outside your documents:

     ```
     Who is the president of Mars?
     ```

     Response:

     ```
     💡 Uncheck to include general knowledge.
     ```

* Technical Note: `graph_rag_query()` now respects the `use_docs_only` flag by omitting fallback contexts outside uploaded documents.

---

## 🗂️ Graph Structure

```
Paper Nodes:
  ┌─────────────────────┐        ┌─────────────────┐
  │ Graph RAG with Neo4j│        │ Intro to RAG    │
  └─────────┬───────────┘        └─────────┬───────┘
            │ RELATED_TO                   │ RELATED_TO
            ▼                               ▼
         ┌─────┐                         ┌─────┐
         │Neo4j│                         │ RAG │
         └─────┘                         └─────┘
            ▲                               ▲
            │ CONNECTED_TO                  │
         ┌─────┐                         ┌─────┐
         │ Topic│                         │Topic│
         └─────┘                         └─────┘
            │
            │ AUTHORED_BY
            ▼
         ┌─────┐
         │Alice│
         └─────┘
```

---

## 🏁 Quick Start Example

1. **Upload documents**

   * PDF or TXT files in Streamlit frontend

2. **Process documents**

   * Chunks are stored in Neo4j and indexed in FAISS

3. **Ask a question**

   * Enter text in the form
   * Enable **“Use only uploaded documents”** to restrict answers

4. **Clear documents (optional)**

   * Click **“Clear Uploaded Documents”**
   * Confirm deletion

**Example Session**:

| Step | Action                             | Result                                                  |
| ---- | ---------------------------------- | ------------------------------------------------------- |
| 1    | Upload `neo4j_intro.pdf`           | ✅ File saved to `uploads/`                              |
| 2    | Process documents                  | ✅ Chunks stored in Neo4j + FAISS built                  |
| 3    | Ask “Where was Barack Obama born?” | If Docs-Only: no answer unless covered in uploaded docs |
| 4    | Disable Docs-Only                  | Answer may include general knowledge                    |
| 5    | Clear documents                    | ✅ Files deleted, vectorstore reset                      |

---

## 📝 Notes

* `config.py`:

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j123"))
embeddings = ...  # your embeddings model
llm = ...         # your LLM model
```

* `graph_demo.py` is safe to run multiple times — idempotent
* Neo4j is the primary focus; use the Browser to visualize nodes/relationships

---

## 🧠 References

* [Neo4j Documentation](https://neo4j.com/docs/)
* [LangChain](https://www.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)

---

## 👩‍💻 Author
### Pranavi Kolipaka
Feel free to connect: 
- [LinkedIn] (https://www.linkedin.com/in/vns-sai-pranavi-kolipaka-489601208/) 
- [GitHub] (https://github.com/Pranavi2002)