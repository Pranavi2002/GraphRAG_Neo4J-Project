# Graph + Vector RAG System (Neo4j-Focused)

This project demonstrates a **Retrieval-Augmented Generation (RAG) system** with a strong focus on **Neo4j graph database** for storing, retrieving, and querying document chunks and metadata. It also integrates **FAISS** for vector similarity search and a **Streamlit frontend** for interacting with the system.

---

## 📂 Folder Structure

```
graph_rag_project/
├── config.py                  # Configuration file (Neo4j credentials, embeddings, LLM) – NOT included in GitHub
├── graph_demo.py              # Neo4j demo: create graph nodes (Paper, Topic, Author) & relationships
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

## ⚙️ Neo4j Setup (Main Focus)

1. **Install Neo4j Desktop**: [Download](https://neo4j.com/download/)
2. **Create a Local Database**:

   * Add → Local DBMS
   * Name: `graph_rag_local`
   * Password: e.g., `neo4j123`
   * Version: Neo4j 5.x (default)
3. **Start the Database**

   * Bolt URL: `bolt://localhost:7687`
   * Browser URL: `http://localhost:7474`

---

## 🏗️ Using Neo4j with Python

### 1️⃣ Create Graph (graph_demo.py)

* Creates nodes:

  * `Paper` → research papers
  * `Topic` → topics
  * `Author` → authors
* Creates relationships:

  * `RELATED_TO` → links Paper ↔ Topic
  * `AUTHORED_BY` → links Paper ↔ Author
  * `CONNECTED_TO` → links Topic ↔ Topic

**Example Multi-Hop Query**:

```python
def multi_hop_query(tx, target_topic):
    """
    Finds papers related to topics connected to a given topic.
    """
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

You will see:

```
✅ Graph data created successfully!
🔍 Retrieved context for topic 'Neo4j':
Title: Graph RAG with Neo4j | Summary: Using graphs to improve retrieval
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
MATCH (n)
DETACH DELETE n
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

  * Load PDFs/TXT files from `uploads/`
  * Split into chunks using `RecursiveCharacterTextSplitter`
  * Store chunks as `Chunk` nodes in Neo4j linked to `Document` nodes
  * Use Neo4j for **graph-based retrieval** and FAISS for **vector-based retrieval**
  * Combine both contexts for LLM-based answers

---

### 4️⃣ Streamlit Frontend

* Upload documents and process them (Neo4j + FAISS)
* Ask questions about uploaded documents
* Option to **use only documents** or include general knowledge
* Clear uploaded documents with confirmation

```bash
streamlit run streamlit_app.py
```

---

## 📝 Notes

* `config.py` contains:

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j123"))
embeddings = ...  # your embeddings model
llm = ...         # your LLM model
```

* Run `graph_demo.py` anytime — safe, idempotent, avoids duplicates.
* Neo4j is the primary focus — use the Browser to visualize and verify nodes/relationships.

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