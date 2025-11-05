from neo4j import GraphDatabase
from config import driver

# # ---------------------------
# # ⚡ Connect to local Neo4j
# # ---------------------------
# uri = "bolt://localhost:7687"
# user = "neo4j"
# password = "neo4j123"   # Use your Neo4j password here

# driver = GraphDatabase.driver(uri, auth=(user, password))

# ---------------------------
# 1️⃣ Create initial graph
# ---------------------------
def create_graph(tx):
    """
    Creates initial papers, topics, authors, and relationships.
    """
    tx.run("""
    CREATE (p1:Paper {title:'Graph RAG with Neo4j', summary:'Using graphs to improve retrieval'})
    CREATE (p2:Paper {title:'Intro to RAG', summary:'Combining retrieval and generation'})
    CREATE (t1:Topic {name:'RAG'})
    CREATE (t2:Topic {name:'Neo4j'})
    CREATE (a1:Author {name:'Alice'})
    CREATE (a2:Author {name:'Bob'})
    CREATE (p1)-[:RELATED_TO]->(t2)
    CREATE (p2)-[:RELATED_TO]->(t1)
    CREATE (p1)-[:AUTHORED_BY]->(a1)
    CREATE (p2)-[:AUTHORED_BY]->(a2)
    CREATE (t1)-[:CONNECTED_TO]->(t2)
    """)

# ---------------------------
# 2️⃣ Retrieve context by topic
# ---------------------------
def retrieve_context(tx, topic):
    """
    Retrieves all papers related to a specific topic (no duplicates).
    """
    query = """
    MATCH (p:Paper)-[:RELATED_TO]->(t:Topic {name: $topic})
    RETURN DISTINCT p.title AS title, p.summary AS summary
    """
    result = tx.run(query, topic=topic)
    return [{"title": record["title"], "summary": record["summary"]} for record in result]

# ---------------------------
# 3️⃣ Add topic safely and link to a paper
# ---------------------------
def add_topic_and_link(tx, paper_title, topic_name):
    """
    Adds a topic if it doesn't exist and links it to a paper.
    Uses MERGE to avoid duplicates. Fixed for Neo4j 5 syntax.
    """
    tx.run("""
    MERGE (t:Topic {name: $topic_name})
    WITH t
    MATCH (p:Paper {title: $paper_title})
    MERGE (p)-[:RELATED_TO]->(t)
    """, paper_title=paper_title, topic_name=topic_name)

# ---------------------------
# 4️⃣ Add author safely and link to a paper
# ---------------------------
def add_author_and_link(tx, paper_title, author_name):
    """
    Adds an author if it doesn't exist and links them to a paper.
    Uses MERGE to avoid duplicates. Fixed for Neo4j 5 syntax.
    """
    tx.run("""
    MERGE (a:Author {name: $author_name})
    WITH a
    MATCH (p:Paper {title: $paper_title})
    MERGE (p)-[:AUTHORED_BY]->(a)
    """, paper_title=paper_title, author_name=author_name)

# ---------------------------
# 5️⃣ Multi-hop query example
# ---------------------------
def multi_hop_query(tx, target_topic):
    """
    Finds papers related to topics that are connected to a given topic (no duplicates).
    Demonstrates multi-hop traversal.
    """
    query = """
    MATCH (p:Paper)-[:RELATED_TO]->(:Topic)-[:CONNECTED_TO]->(t:Topic {name: $target_topic})
    RETURN DISTINCT p.title AS title
    """
    result = tx.run(query, target_topic=target_topic)
    return [record["title"] for record in result]

# ---------------------------
# 🚀 Main session
# ---------------------------
with driver.session() as session:

    # Create initial graph
    session.execute_write(create_graph)
    print("✅ Graph data created successfully!")

    # Add a new topic "AI" and link to paper
    session.execute_write(add_topic_and_link, "Graph RAG with Neo4j", "AI")
    print("✅ New topic 'AI' added and linked to 'Graph RAG with Neo4j'")

    # Add a new author "Charlie" and link to paper
    session.execute_write(add_author_and_link, "Intro to RAG", "Charlie")
    print("✅ New author 'Charlie' added and linked to 'Intro to RAG'")

    # Retrieve context for topic "Neo4j"
    context = session.execute_read(retrieve_context, "Neo4j")
    print("\n🔍 Retrieved context for topic 'Neo4j':")
    for c in context:
        print(f"Title: {c['title']} | Summary: {c['summary']}")

    # Run multi-hop query for topic "Neo4j"
    multi_hop_results = session.execute_read(multi_hop_query, "Neo4j")
    print("\n🔍 Multi-hop query results for topic 'Neo4j':")
    for title in multi_hop_results:
        print(title)

# Close the driver
driver.close()