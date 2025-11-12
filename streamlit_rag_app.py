import os
import streamlit as st
import rag_app_streamlit as rag

# ---------------------------------
# Setup
# ---------------------------------
os.makedirs("uploads", exist_ok=True)
st.set_page_config(page_title="Plain RAG (Docs Only)", page_icon="📄", layout="centered")
st.title("📄 Plain RAG System (Strict Docs-Only)")

# ---------------------------------
# Session state for file list
# ---------------------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = os.listdir("uploads")

def refresh_file_list():
    st.session_state.uploaded_files = os.listdir("uploads")

# ---------------------------------
# Clear all uploaded documents
# ---------------------------------
# Initialize session state
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

# Step 1: Show delete button
if st.button("🗑️ Clear Uploaded Documents"):
    st.session_state.confirm_delete = True

# Step 2: Show confirmation checkbox if button was clicked
if st.session_state.confirm_delete:
    st.warning("⚠️ This will delete all uploaded documents permanently!")
    confirm = st.checkbox("Yes, I want to delete all uploaded documents")
    if confirm:
        try:
            for f in os.listdir("uploads"):
                os.remove(os.path.join("uploads", f))
            vectorstore = None  # reset vectorstore
            st.success("✅ All uploaded documents cleared!")
            st.session_state.confirm_delete = False  # hide confirmation
        except Exception as e:
            st.error(f"⚠️ Could not clear documents: {e}")

# ---------------------------------
# Uploaded Documents List with Delete Buttons
# ---------------------------------
if st.session_state.uploaded_files:
    st.subheader("📂 Uploaded Documents:")
    updated_files = st.session_state.uploaded_files.copy()
    for f in st.session_state.uploaded_files:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.write(f"- {f}")
        with col2:
            if st.button(f"Delete {f}", key=f"delete_{f}"):
                try:
                    os.remove(os.path.join("uploads", f))
                    rag.vectorstore = None
                    updated_files.remove(f)
                    st.success(f"✅ Document '{f}' deleted")
                    st.session_state.uploaded_files = updated_files
                except Exception as e:
                    st.error(f"⚠️ Could not delete document '{f}': {e}")
else:
    st.info("No documents uploaded yet.")

st.divider()

# ---------------------------------
# Upload documents
# ---------------------------------
uploaded_files = st.file_uploader(
    "📤 Upload PDFs or Text files",
    type=["pdf", "txt"],
    accept_multiple_files=True
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
    refresh_file_list()  # update session state

# ---------------------------------
# Process documents if needed
# ---------------------------------
if new_files or rag.vectorstore is None:
    try:
        docs = rag.load_documents("uploads")
        if docs:
            with st.spinner("⏳ Processing documents..."):
                chunks = rag.split_documents(docs)
                rag.build_vectorstore(chunks)
            st.success(f"✅ Processed {len(chunks)} chunks and built vectorstore")
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
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    submitted = st.form_submit_button("Get Answer")

if submitted and question:
    with st.spinner("⏳ Processing your question..."):
        try:
            answer = rag.rag_query_strict(question, k=3)
            st.subheader("💡 Answer:")
            st.write(answer)

            if show_chunks and rag.vectorstore is not None:
                docs = rag.vectorstore.similarity_search(question, k=3)
                if docs:
                    st.subheader("📄 Retrieved Chunks (Top 3):")
                    for i, d in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}:** {d.page_content[:300]}...")
                else:
                    st.info("No chunks retrieved for this question.")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")