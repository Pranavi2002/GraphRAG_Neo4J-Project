# no dynamic hop input for multi-hop traversal and no visualization of the traversed graph paths

import os
import streamlit as st
import graph_rag_app_streamlit as rag  # backend module

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

st.title("📚 Graph + Vector RAG System")

# ---------------------------
# Clear uploaded documents with two-step confirmation (includes Neo4j + local)
# ---------------------------
# Initialize session state
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

# Step 1: Show delete button
if st.button("🗑️ Clear Uploaded Documents (Neo4j + Local)"):
    st.session_state.confirm_delete = True

# Step 2: Show confirmation checkbox if button was clicked
if st.session_state.confirm_delete:
    st.warning("⚠️ This will delete all uploaded documents and Neo4j data permanently!")
    confirm = st.checkbox("Yes, I want to delete all uploaded documents and graph data")

    if confirm:
        try:
            # 🗑️ 1️⃣ Delete files from local 'uploads' folder
            for f in os.listdir("uploads"):
                os.remove(os.path.join("uploads", f))

            # 🗑️ 2️⃣ Delete all docs/entities from Neo4j
            rag.delete_all_docs()

            # 🗑️ 3️⃣ Reset in-memory FAISS vectorstore
            rag.vectorstore = None

            st.success("✅ All uploaded documents and Neo4j data cleared!")
            st.session_state.confirm_delete = False  # hide confirmation
        except Exception as e:
            st.error(f"⚠️ Could not clear documents: {e}")

# ---------------------------
# 1️⃣ Upload documents
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs or Text files", type=["pdf", "txt"], accept_multiple_files=True
)

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

# ---------------------------
# 1️⃣a Show list of all uploaded files
# ---------------------------
all_files = os.listdir("uploads")
if all_files:
    st.subheader("📂 Uploaded Documents:")
    for f in all_files:
        st.write(f"- {f}")
else:
    st.info("No documents uploaded yet.")

# ---------------------------
# 2️⃣ Process documents (existing + new)
# ---------------------------
if new_files or rag.vectorstore is None:
    try:
        docs = rag.load_documents("uploads")
        if docs:  # Only process if there are documents
            with st.spinner("⏳ Processing documents (splitting, entity extraction, graph + vector build)..."):
                chunks = rag.split_documents(docs)
                rag.store_in_neo4j(chunks)  # includes LLM refinement
                rag.build_vectorstore(chunks)
            st.success(f"✅ Processed {len(chunks)} document chunks and built/updated vectorstore")
        else:
            st.info("No documents to process.")
    except Exception as e:
        st.error(f"⚠️ Error processing documents: {e}")

# Initialize session state for processing
if "processing_query" not in st.session_state:
    st.session_state.processing_query = False

# ---------------------------
# 3️⃣ Ask a question (with Enter submission)
# ---------------------------
with st.form("question_form"):
    # Disable textbox while processing
    question = st.text_input(
        "Ask a question about your documents:",
        disabled=st.session_state.processing_query
    )
    use_docs_only = st.checkbox(
        "Use only uploaded documents (no general knowledge)", value=True
    )
    submitted = st.form_submit_button("Get Answer")

    if submitted and question:
        try:
            # Mark as processing
            st.session_state.processing_query = True
            # Show spinner while processing
            with st.spinner("⏳ Processing your question..."):
                answer = rag.graph_rag_query(question, use_docs_only=use_docs_only)
            st.subheader("💡 Answer:")
            st.write(answer)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
        finally:
            # Reset processing flag so textbox is enabled again
            st.session_state.processing_query = False