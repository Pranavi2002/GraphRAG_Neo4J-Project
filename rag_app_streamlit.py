import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import embeddings, llm

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
# 2️⃣ Build FAISS vectorstore
# ---------------------------
def build_vectorstore(chunks):
    global vectorstore
    print("🔧 Building vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vectorstore built successfully")
    return vectorstore


# ---------------------------
# 3️⃣ Strict docs-only RAG query
# ---------------------------
def rag_query_strict(question, k=3):
    """
    Returns an answer strictly based on uploaded documents.
    If no relevant chunks found, returns a warning.
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("Vectorstore not built yet!")

    # Retrieve top-k relevant chunks using vectorstore directly
    docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return f"⚠️ No information about '{question}' found in the uploaded documents."

    # Concatenate chunk texts
    context = "\n\n".join([d.page_content for d in docs])

    # Construct prompt
    prompt = f"""
You are a helpful assistant.
Answer strictly based on the below context only.
If the context does not contain the answer, say "Information not found in uploaded documents."

Context:
{context}

Question: {question}
Answer:
"""

    # Generate answer using LLM
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content