# no delete a doc feature, no colors for nodes

import os
import streamlit as st
import graph_rag_app_streamlit as rag
from pyvis.network import Network
import networkx as nx

# ---------------------------------
# Setup
# ---------------------------------
os.makedirs("uploads", exist_ok=True)
st.set_page_config(page_title="Graph + Vector RAG", page_icon="📚", layout="centered")
st.title("📚 Graph + Vector RAG System")

# ---------------------------------
# Clear uploaded documents
# ---------------------------------
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

if st.button("🗑️ Clear Uploaded Documents (Neo4j + Local)"):
    st.session_state.confirm_delete = True

if st.session_state.confirm_delete:
    st.warning("⚠️ This will delete all uploaded documents and Neo4j data permanently!")
    confirm = st.checkbox("Yes, I want to delete all uploaded documents and graph data")

    if confirm:
        try:
            for f in os.listdir("uploads"):
                os.remove(os.path.join("uploads", f))
            rag.delete_all_docs()
            rag.vectorstore = None
            st.success("✅ All uploaded documents and Neo4j data cleared!")
            st.session_state.confirm_delete = False
        except Exception as e:
            st.error(f"⚠️ Could not clear documents: {e}")

st.divider()

# ---------------------------------
# Upload and process documents
# ---------------------------------
uploaded_files = st.file_uploader("📤 Upload PDFs or Text files", type=["pdf", "txt"], accept_multiple_files=True)
new_files = []
if uploaded_files:
    for file in uploaded_files:
        save_path = f"uploads/{file.name}"
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            new_files.append(file.name)

if new_files:
    st.success(f"✅ Uploaded {len(new_files)} new files: {', '.join(new_files)}")

all_files = os.listdir("uploads")
if all_files:
    st.subheader("📂 Uploaded Documents:")
    for f in all_files:
        st.write(f"- {f}")
else:
    st.info("No documents uploaded yet.")

# Process documents
if new_files or rag.vectorstore is None:
    try:
        docs = rag.load_documents("uploads")
        if docs:
            with st.spinner("⏳ Processing documents..."):
                chunks = rag.split_documents(docs)
                rag.store_in_neo4j(chunks)
                rag.build_vectorstore(chunks)
            st.success(f"✅ Processed {len(chunks)} document chunks and built/updated vectorstore")
        else:
            st.info("No documents to process.")
    except Exception as e:
        st.error(f"⚠️ Error processing documents: {e}")

st.divider()

# ---------------------------------
# Question form
# ---------------------------------
with st.form("question_form"):
    st.subheader("💬 Ask a Question")
    question = st.text_input("Ask a question about your documents:")
    hops = st.slider("Select number of hops (graph traversal depth)", min_value=1, max_value=5, value=3)
    use_docs_only = st.checkbox("Use only uploaded documents (no general knowledge)", value=True)
    show_paths = st.checkbox("Show traversed graph paths", value=False)
    submitted = st.form_submit_button("Get Answer")

if submitted and question:
    with st.spinner("⏳ Processing your question..."):
        try:
            # ---------------------------
            # Get answer from RAG
            # ---------------------------
            answer = rag.graph_rag_query(question, hops=hops, use_docs_only=use_docs_only)
            st.subheader("💡 Answer:")
            st.write(answer)

            if show_paths:
                st.subheader("🔍 Traversed Graph Visualization")

                # ---------------------------
                # Extract entities from question (simple approach)
                # ---------------------------
                # For now, just split question words and filter for capitalized words (entity candidates)
                entity_candidates = [w for w in question.split() if w[0].isupper()]
                if not entity_candidates:
                    entity_candidates = [question]  # fallback

                # ---------------------------
                # Fetch paths from Neo4j
                # ---------------------------
                all_paths = []
                with rag.driver.session() as session:
                    for entity in entity_candidates:
                        query = f"""
                            MATCH path=(start:Entity)-[*1..{hops}]-(related)
                            WHERE ANY(node IN nodes(path) WHERE toLower(node.name) CONTAINS toLower($entity))
                            RETURN [node IN nodes(path) | node.name] AS path_nodes
                            LIMIT 10
                        """
                        result = session.run(query, {"entity": entity})
                        all_paths.extend([record["path_nodes"] for record in result])

                if all_paths:
                    # ---------------------------
                    # Build NetworkX graph
                    # ---------------------------
                    G = nx.Graph()
                    for path in all_paths:
                        # Filter out None or empty nodes
                        path = [node for node in path if node]
                        for i in range(len(path)-1):
                            G.add_edge(path[i], path[i+1])

                    # ---------------------------
                    # Pyvis network
                    # ---------------------------
                    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
                    net.from_nx(G)
                    net.show_buttons(filter_=['physics'])

                    # Save and embed in Streamlit
                    tmp_path = "graph.html"
                    net.save_graph(tmp_path)
                    st.components.v1.html(open(tmp_path, 'r', encoding='utf-8').read(), height=550)
                else:
                    st.info("No paths found for the current question.")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")