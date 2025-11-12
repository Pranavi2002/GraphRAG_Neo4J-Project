# hallucinates or uses memory to generate answer
# no retrieved chunks and has delete file dropdown

import os, streamlit as st
import rag_app_streamlit1 as rag

# ---------------------------------
# Setup
# ---------------------------------
os.makedirs("uploads", exist_ok=True)
st.set_page_config(page_title="Plain RAG", page_icon="📄", layout="centered")
st.title("📄 Plain RAG System")

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
# Delete a specific document
# ---------------------------------
docs_list = os.listdir("uploads")
if docs_list:
    doc_to_delete = st.selectbox("Select document to delete:", docs_list)
    if st.button(f"Delete '{doc_to_delete}'"):
        try:
            rag.delete_doc(doc_to_delete)
            st.success(f"✅ Document '{doc_to_delete}' deleted")
        except Exception as e:
            st.error(f"⚠️ Could not delete document: {e}")
else:
    st.info("No documents available for deletion.")

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
    use_docs_only = st.checkbox("Use only uploaded documents", value=True)
    submitted = st.form_submit_button("Get Answer")

if submitted and question:
    with st.spinner("⏳ Processing your question..."):
        try:
            answer = rag.rag_query(question, use_docs_only=use_docs_only)
            st.subheader("💡 Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")